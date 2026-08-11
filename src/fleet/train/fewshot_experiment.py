#!/usr/bin/env python3
"""
Few-shot Learning with Real Avoidance
Improved version:
1. Configurable pretrain checkpoint path
2. During training, select 10 real images from cc12m-2mp-realistic for contrastive learning
3. Avoidance loss: real samples also avoid the fake attention weights
"""

import os
import json
import argparse
import glob
import random
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fleet.utils import (
    resolve_dinov3_model_path,
    coalesce_cli_env,
    infer_q_dim_and_spec_from_paths,
    load_q_train_module,
    fewshot_contrastive_loss as contrastive_loss,
    distillation_loss,
    mutual_avoidance_loss,
)
from fleet.datasets import collect_aigibench_image_paths
from fleet.datasets.dual_branch_fewshot import DualBranchImageDataset


# ==================== Anchor computation ====================
def compute_anchors(model, memory_loader, device):
    """
    Compute fake and real attention-weight anchors from the replay set (memory set).

    Args:
        model: the model
        memory_loader: memory-set data loader
        device: device

    Returns:
        fake_anchor: [num_heads] - mean attention weights of fake samples
        real_anchor: [num_heads] - mean attention weights of real samples
    """
    model.eval()

    all_fake_attn = []
    all_real_attn = []

    print("  Computing attention-weight anchors from the replay set...")

    with torch.no_grad():
        for img_dinov3, img_xception, labels in memory_loader:
            img_dinov3 = img_dinov3.to(device, non_blocking=True)
            img_xception = img_xception.to(device, non_blocking=True)
            labels = labels.to(device)

            if isinstance(model, nn.DataParallel):
                _, attn_weights = model(img_dinov3, img_xception)
            else:
                _, attn_weights = model(img_dinov3, img_xception)

            # Separate fake and real
            fake_mask = (labels == 0)
            real_mask = (labels == 1)

            if fake_mask.sum() > 0:
                all_fake_attn.append(attn_weights[fake_mask].cpu())
            if real_mask.sum() > 0:
                all_real_attn.append(attn_weights[real_mask].cpu())

    # Compute the mean
    all_fake_attn = torch.cat(all_fake_attn, dim=0)  # [n_fake, num_heads]
    all_real_attn = torch.cat(all_real_attn, dim=0)  # [n_real, num_heads]

    fake_anchor = all_fake_attn.mean(dim=0)  # [num_heads]
    real_anchor = all_real_attn.mean(dim=0)  # [num_heads]

    anchor_dot = torch.sum(fake_anchor * real_anchor)
    anchor_cosine = anchor_dot / (torch.norm(fake_anchor) * torch.norm(real_anchor))
    print(f"  Anchor stats: dot={anchor_dot.item():.4f}, cosine={anchor_cosine.item():.4f}")

    return fake_anchor.to(device), real_anchor.to(device)


# ==================== Training function ====================
def train_epoch(model, support_loader, memory_loader, optimizer, device,
                memory_features_dict, fake_anchor, real_anchor,
                avoid_weight=1.0, distill_weight=1.0, contrastive_weight=1.0, temperature=0.07):
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    total_c_loss = 0.0
    total_distill_loss = 0.0
    total_avoid_loss = 0.0
    n_batches = 0

    # Gather all support data
    support_images_dinov3_all = []
    support_images_xception_all = []
    support_labels_all = []

    for img_dinov3, img_xception, label in support_loader:
        support_images_dinov3_all.append(img_dinov3)
        support_images_xception_all.append(img_xception)
        support_labels_all.append(label)

    if len(support_images_dinov3_all) == 0:
        return 0, 0, 0, 0

    support_images_dinov3_all = torch.cat(support_images_dinov3_all, dim=0).to(device, non_blocking=True)
    support_images_xception_all = torch.cat(support_images_xception_all, dim=0).to(device, non_blocking=True)
    support_labels_all = torch.cat(support_labels_all, dim=0).to(device)

    # Iterate over the memory data
    for batch_idx, (memory_images_dinov3, memory_images_xception, memory_labels) in enumerate(memory_loader):
        memory_images_dinov3 = memory_images_dinov3.to(device, non_blocking=True)
        memory_images_xception = memory_images_xception.to(device, non_blocking=True)
        memory_labels = memory_labels.to(device)

        # Merge batches
        images_dinov3 = torch.cat([support_images_dinov3_all, memory_images_dinov3], dim=0)
        images_xception = torch.cat([support_images_xception_all, memory_images_xception], dim=0)
        labels = torch.cat([support_labels_all, memory_labels], dim=0)

        if isinstance(model, nn.DataParallel):
            features, attn_weights = model(images_dinov3, images_xception)
        else:
            features, attn_weights = model(images_dinov3, images_xception)

        # Contrastive loss
        c_loss = contrastive_loss(features, labels, temperature)

        # Distillation loss
        distill_loss_val = torch.tensor(0.0, device=device)
        if memory_features_dict is not None and batch_idx in memory_features_dict:
            n_support = support_images_dinov3_all.size(0)
            memory_features_current = features[n_support:]
            memory_features_original = memory_features_dict[batch_idx].to(device)
            distill_loss_val = distillation_loss(memory_features_current, memory_features_original, distill_weight)

        # Mutual avoidance loss (support samples avoid the replay-set anchors)
        n_support = support_images_dinov3_all.size(0)
        support_attn_weights = attn_weights[:n_support]
        support_labels = labels[:n_support]
        avoid_loss_val = mutual_avoidance_loss(support_attn_weights, support_labels,
                                              fake_anchor, real_anchor, avoid_weight)

        # Total loss (apply the contrastive-loss weight)
        loss = c_loss * contrastive_weight + distill_loss_val + avoid_loss_val

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_c_loss += c_loss.item()
        total_distill_loss += distill_loss_val.item()
        total_avoid_loss += avoid_loss_val.item()
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    avg_c_loss = total_c_loss / n_batches if n_batches > 0 else 0
    avg_distill_loss = total_distill_loss / n_batches if n_batches > 0 else 0
    avg_avoid_loss = total_avoid_loss / n_batches if n_batches > 0 else 0

    return avg_loss, avg_c_loss, avg_distill_loss, avg_avoid_loss


# ==================== Validation function ====================
def validate_with_prototypes(model, validation_sets, fake_prototype, real_prototype,
                            processor, device, xception_crop_size=128,
                            batch_size=256, num_workers=8):
    """
    Validate using prototype vectors and save the per-image confidence.

    Confidence: S(x) = CosSim(f(x), P_pos) - CosSim(f(x), P_neg)
    where P_pos = real_prototype, P_neg = fake_prototype
    """
    model.eval()

    results = {}

    for dataset_name, dataset_info in validation_sets.items():
        paths = dataset_info['paths']
        labels = dataset_info['labels']

        if len(paths) == 0:
            results[dataset_name] = {
                'total_acc': 0.0,
                'fake_acc': 0.0,
                'real_acc': 0.0,
                'n_samples': 0,
                'confidences': [],
                'confidence_paths': []
            }
            continue

        dataset = DualBranchImageDataset(paths, labels, processor, xception_crop_size, is_training=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for img_dinov3, img_xception, batch_labels in dataloader:
                img_dinov3 = img_dinov3.to(device, non_blocking=True)
                img_xception = img_xception.to(device, non_blocking=True)

                if isinstance(model, nn.DataParallel):
                    features, _ = model(img_dinov3, img_xception)
                else:
                    features, _ = model(img_dinov3, img_xception)

                all_features.append(features.cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)

        # Compute cosine similarity
        fake_prototype_np = fake_prototype.cpu().numpy()
        real_prototype_np = real_prototype.cpu().numpy()

        # Normalize features and prototypes
        all_features_norm = all_features / (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8)
        fake_prototype_norm = fake_prototype_np / (np.linalg.norm(fake_prototype_np) + 1e-8)
        real_prototype_norm = real_prototype_np / (np.linalg.norm(real_prototype_np) + 1e-8)

        # Compute cosine similarity
        fake_sim = np.dot(all_features_norm, fake_prototype_norm)
        real_sim = np.dot(all_features_norm, real_prototype_norm)

        # Compute confidence: S(x) = CosSim(f(x), P_pos) - CosSim(f(x), P_neg)
        # P_pos = real_prototype, P_neg = fake_prototype
        confidences = real_sim - fake_sim

        # Predict using the confidence (S(x) > 0 -> positive class)
        predictions = (confidences > 0).astype(int)

        total_acc = (predictions == all_labels).mean() * 100

        fake_mask = all_labels == 0
        real_mask = all_labels == 1

        fake_acc = 0.0
        real_acc = 0.0
        if fake_mask.sum() > 0:
            fake_acc = (predictions[fake_mask] == all_labels[fake_mask]).mean() * 100
        if real_mask.sum() > 0:
            real_acc = (predictions[real_mask] == all_labels[real_mask]).mean() * 100

        # Compute per-class accuracy
        results[dataset_name] = {
            'total_acc': float(total_acc),
            'fake_acc': float(fake_acc),
            'real_acc': float(real_acc),
            'n_samples': len(paths),
            'confidences': confidences.tolist(),  # save per-image confidence
            'confidence_paths': paths  # save the corresponding image paths
        }

    return results


def print_validation_summary(title, results):
    print(title)
    for dataset_name, result in results.items():
        print(
            f"  {dataset_name}: total={result['total_acc']:.2f}% "
            f"fake={result['fake_acc']:.2f}% real={result['real_acc']:.2f}%"
        )


# ==================== Main ====================
def run_experiment(args):
    """Run the experiment."""
    print(f"[Few-shot] Start: {args.dataset_name}")

    exp_start_time = time.time()

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}, visible GPUs: {torch.cuda.device_count()}")

    # Load the DINOv3 processor
    print("\nLoading DINOv3 image processor...")
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(args.dinov3_model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: failed to load DINOv3 processor: {e}")
        processor = None

    # ========== Infer the model structure from the weight/prototype directories ==========
    inferred_q_dim, inferred_q_spec = infer_q_dim_and_spec_from_paths(
        [args.prototype_dir, args.pretrain_checkpoint]
    )
    if inferred_q_dim is None:
        raise RuntimeError("Cannot infer q_dim from the paths.")
    if inferred_q_spec is None:
        inferred_q_spec = str(inferred_q_dim)
    if inferred_q_dim <= 0 or 1024 % inferred_q_dim != 0:
        raise ValueError(f"q_dim must be a divisor of 1024, got q_dim={inferred_q_dim}")
    num_heads = 1024 // inferred_q_dim

    q_mod = load_q_train_module(inferred_q_spec, fallback_q_dim=inferred_q_dim)
    if not hasattr(q_mod, "DualBranchContrastiveModel"):
        raise RuntimeError(f"{q_mod.__name__} is missing DualBranchContrastiveModel, cannot be used for few-shot.")

    # Load the pretrained model
    print(f"\nLoading pretrained model: {args.pretrain_checkpoint}")
    checkpoint = torch.load(args.pretrain_checkpoint, map_location=device, weights_only=False)
    # Feature config is read automatically from the checkpoint (consistent with pretraining)
    use_lhs = bool(checkpoint.get('use_last_hidden_state', False))
    norm_feat = bool(checkpoint.get('normalize_feature', False))
    feat_layer = checkpoint.get('feature_layer', -1)
    model = q_mod.DualBranchContrastiveModel(
        dinov3_model_path=args.dinov3_model_path,
        xception_feature_dim=args.xception_feature_dim,
        projection_dim=args.projection_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_heads=num_heads,
        q_dim=inferred_q_dim,
        feature_layer=feat_layer,
        use_last_hidden_state=use_lhs,
        normalize_feature=norm_feat,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if num_gpus >= 8:
            device_ids = list(range(8))
            print(f"Using all 8 GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            print(f"Using {num_gpus} GPUs (fewer than 8 available)")
            model = nn.DataParallel(model)
    else:
        print("Warning: using only 1 GPU, 8 GPUs are recommended for best performance")

    if isinstance(model, nn.DataParallel):
        trainable_params = []
        for _, lora_layer in model.module.lora_modules:
            trainable_params.append(lora_layer.lora_A)
            trainable_params.append(lora_layer.lora_B)
        trainable_params.extend(model.module.xception.parameters())
        trainable_params.extend(model.module.xception_fc.parameters())
        trainable_params.extend(model.module.routing.parameters())
        trainable_params.extend(model.module.projection.parameters())
    else:
        trainable_params = []
        for _, lora_layer in model.lora_modules:
            trainable_params.append(lora_layer.lora_A)
            trainable_params.append(lora_layer.lora_B)
        trainable_params.extend(model.xception.parameters())
        trainable_params.extend(model.xception_fc.parameters())
        trainable_params.extend(model.routing.parameters())
        trainable_params.extend(model.projection.parameters())

    trainable_params = [p for p in trainable_params if p.requires_grad]

    # Load the prototype vectors
    print(f"\nLoading prototype vectors: {args.prototype_dir}")
    fake_prototype_path = os.path.join(args.prototype_dir, 'fake_prototype.npy')
    real_prototype_path = os.path.join(args.prototype_dir, 'real_prototype.npy')

    fake_prototype = torch.from_numpy(np.load(fake_prototype_path)).to(device)
    real_prototype = torch.from_numpy(np.load(real_prototype_path)).to(device)
    print(f"  Prototype vectors loaded: Fake {fake_prototype.shape}, Real {real_prototype.shape}")

    # Prepare the data
    print("\nPreparing data...")

    # 1. Support set
    print("  Preparing Support set...")
    query_fake_dir = args.query_dir
    query_fake_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        query_fake_paths.extend(glob.glob(os.path.join(query_fake_dir, ext)))
        query_fake_paths.extend(glob.glob(os.path.join(query_fake_dir, ext.upper())))

    if len(query_fake_paths) < args.n_fake_support:
        print(f"Error: Query directory has fewer than {args.n_fake_support} images ({len(query_fake_paths)})")
        return

    random.seed(42)
    support_fake_paths = random.sample(query_fake_paths, args.n_fake_support)

    # real support directory: reuse real_val_dir and filter images whose filename contains cc12m
    real_support_dir = args.real_val_dir
    real_support_paths_all = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        real_support_paths_all.extend(glob.glob(os.path.join(real_support_dir, '**', ext), recursive=True))
        real_support_paths_all.extend(glob.glob(os.path.join(real_support_dir, '**', ext.upper()), recursive=True))
    real_support_paths_all = [p for p in real_support_paths_all if 'cc12m' in os.path.basename(p).lower()]

    if len(real_support_paths_all) < args.n_real_support:
        print(f"Error: real_support has fewer than {args.n_real_support} images ({len(real_support_paths_all)}) - {real_support_dir} (filename contains cc12m)")
        return

    support_real_paths = random.sample(real_support_paths_all, args.n_real_support)

    support_paths = support_fake_paths + support_real_paths
    support_labels = [0] * args.n_fake_support + [1] * args.n_real_support

    support_manifest = {
        'dataset_name': args.dataset_name,
        'strategy': 'fake_random_seed42 + real_random_from_real_support_dir_seed42',
        'fake_support_paths': support_fake_paths,
        'real_support_paths': support_real_paths,
        'real_support_dir': real_support_dir,
    }
    support_manifest_path = os.path.join(args.output_dir, 'support_set_paths.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(support_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(support_manifest, f, indent=2, ensure_ascii=False)
    print(f"    Support set: {args.n_fake_support} fake + {args.n_real_support} real = {len(support_paths)} images")

    # 2. Memory set: AIGIBench training set
    print("  Preparing Memory set...")
    train_dir = args.aigibench_train_dir
    if not train_dir:
        raise SystemExit(
            "Few-shot Memory set requires the AIGIBench training-set directory: "
            "specify it via --aigibench_train_dir, or set the FLEET_AIGIBENCH_TRAIN env var"
        )
    train_paths, train_labels = collect_aigibench_image_paths(train_dir)
    memory_fake_paths = [p for p, y in zip(train_paths, train_labels) if y == 0]
    memory_real_paths = [p for p, y in zip(train_paths, train_labels) if y == 1]

    # Sample the specified number of memory-set images
    if len(memory_fake_paths) > args.n_fake_memory:
        memory_fake_paths = random.sample(memory_fake_paths, args.n_fake_memory)
    if len(memory_real_paths) > args.n_real_memory:
        memory_real_paths = random.sample(memory_real_paths, args.n_real_memory)

    # Shuffle memory_paths and memory_labels together (keep them aligned)
    memory_paths = memory_fake_paths + memory_real_paths
    memory_labels = [0] * len(memory_fake_paths) + [1] * len(memory_real_paths)
    combined_memory = list(zip(memory_paths, memory_labels))
    random.shuffle(combined_memory)
    memory_paths, memory_labels = map(list, zip(*combined_memory)) if combined_memory else ([], [])

    # When the support set is 1-shot (one image per class), shuffle the Memory set in advance
    # to avoid bias from a fixed order
    if args.n_fake_support == 1 and args.n_real_support == 1:
        combined = list(zip(memory_paths, memory_labels))
        random.shuffle(combined)
        memory_paths, memory_labels = map(list, zip(*combined))

    print(f"    Memory set: {len(memory_fake_paths)} fake + {len(memory_real_paths)} real = {len(memory_paths)} images")

    # 3. Query set: remaining fake images
    query_paths = [p for p in query_fake_paths if p not in support_fake_paths]
    if len(query_paths) > 1000:
        query_paths = random.sample(query_paths, 1000)
    query_labels = [0] * len(query_paths)

    print(f"    Query set: {len(query_paths)} images")

    # 4. Validation set
    print("  Preparing validation set...")

    # AIGIBench val
    val_dir = args.aigibench_val_dir
    if not val_dir:
        raise SystemExit(
            "AIGIBench/val path is not configured: specify it via --aigibench_val_dir, "
            "or set the FLEET_AIGIBENCH_VAL env var"
        )
    val_paths, val_labels = collect_aigibench_image_paths(val_dir)
    val_fake_paths = [p for p, y in zip(val_paths, val_labels) if y == 0]
    val_real_paths = [p for p, y in zip(val_paths, val_labels) if y == 1]

    # final/real is used for testing (derived from real_val_dir or real_support_dir, excluding support images)
    if args.real_val_dir is not None:
        # Use the standalone real_val_dir
        real_val_paths_all = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            real_val_paths_all.extend(glob.glob(os.path.join(args.real_val_dir, '**', ext), recursive=True))
            real_val_paths_all.extend(glob.glob(os.path.join(args.real_val_dir, '**', ext.upper()), recursive=True))
        real_val_paths = real_val_paths_all
    else:
        # Default: exclude the real images used in the support set from real_support_dir
        real_val_paths = [p for p in real_support_paths_all if p not in support_real_paths]

    if args.max_real_val is not None and len(real_val_paths) > args.max_real_val:
        real_val_paths = random.sample(real_val_paths, args.max_real_val)

    # Build the validation-set dict.
    # NOTE: the "Query集" key prefix is matched by the Run.sh summary script, keep it as-is.
    validation_sets = {
        f'Query集({args.dataset_name})': {
            'paths': query_paths,
            'labels': query_labels
        },
        'AIGIBench/val-fake': {
            'paths': val_fake_paths,
            'labels': [0] * len(val_fake_paths)
        },
        'AIGIBench/val-real': {
            'paths': val_real_paths,
            'labels': [1] * len(val_real_paths)
        },
        'final/real': {
            'paths': real_val_paths,
            'labels': [1] * len(real_val_paths)
        },
    }

    print(f"\n  Total validation sets: {len(validation_sets)}")

    # Create datasets and loaders
    support_dataset = DualBranchImageDataset(support_paths, support_labels, processor,
                                            args.xception_crop_size, is_training=True)
    memory_dataset = DualBranchImageDataset(memory_paths, memory_labels, processor,
                                           args.xception_crop_size, is_training=True)

    support_loader = DataLoader(
        support_dataset,
        batch_size=len(support_paths),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # Distillation features are aligned by batch_idx, so memory_loader must use a deterministic
    # order (shuffle=False); otherwise a mismatch would make the distillation loss explode.
    memory_loader = DataLoader(
        memory_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Extract memory features for distillation, and compute the attention-weight anchors
    print("\nPreparing training resources...")
    print("1. Extracting Memory features for distillation...")
    model.eval()
    memory_features_dict = {}

    with torch.no_grad():
        for batch_idx, (img_dinov3, img_xception, _) in enumerate(memory_loader):
            img_dinov3 = img_dinov3.to(device, non_blocking=True)
            img_xception = img_xception.to(device, non_blocking=True)

            if isinstance(model, nn.DataParallel):
                features, _ = model(img_dinov3, img_xception)
            else:
                features, _ = model(img_dinov3, img_xception)

            memory_features_dict[batch_idx] = features.cpu()

    print(f"  Extracted features for {len(memory_features_dict)} batches")

    # Compute the attention-weight anchors
    print("2. Computing attention-weight anchors (from the replay set)...")
    fake_anchor, real_anchor = compute_anchors(model, memory_loader, device)

    # Initial validation (Epoch 0)
    initial_results = validate_with_prototypes(
        model, validation_sets, fake_prototype, real_prototype,
        processor, device, args.xception_crop_size,
        batch_size=args.val_batch_size, num_workers=args.num_workers
    )

    query_key = f'Query集({args.dataset_name})'
    final_real_key = 'final/real'
    print_validation_summary("Validation before training:", initial_results)

    # Check the Query-set accuracy
    query_key = f'Query集({args.dataset_name})'
    initial_query_acc = initial_results[query_key]['total_acc']
    final_real_key = 'final/real'
    initial_real_acc = initial_results[final_real_key]['total_acc'] if final_real_key in initial_results else 0.0
    initial_avg_acc = (initial_query_acc + initial_real_acc) / 2.0

    if (not args.force_train) and initial_query_acc > 98.0:
        print(f"\n  Query-set accuracy ({initial_query_acc:.2f}%) is already above 98%, skipping training")
        print(f"  final/real accuracy: {initial_real_acc:.2f}%")
        print(f"  Real/Fake average accuracy: {initial_avg_acc:.2f}%")
        final_results = initial_results
        training_log = []
        best_avg_acc = initial_avg_acc
    else:
        # Training
        print("Starting training")

        # Configure the optimizer
        if isinstance(model, nn.DataParallel):
            trainable_params = []
            for name, lora_layer in model.module.lora_modules:
                trainable_params.append(lora_layer.lora_A)
                trainable_params.append(lora_layer.lora_B)
            trainable_params.extend(model.module.xception.parameters())
            trainable_params.extend(model.module.xception_fc.parameters())
            trainable_params.extend(model.module.routing.parameters())
            trainable_params.extend(model.module.projection.parameters())
        else:
            trainable_params = []
            for name, lora_layer in model.lora_modules:
                trainable_params.append(lora_layer.lora_A)
                trainable_params.append(lora_layer.lora_B)
            trainable_params.extend(model.xception.parameters())
            trainable_params.extend(model.xception_fc.parameters())
            trainable_params.extend(model.routing.parameters())
            trainable_params.extend(model.projection.parameters())

        trainable_params = [p for p in trainable_params if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=3e-5, weight_decay=0.01)

        training_log = []
        best_avg_acc = initial_avg_acc

        for epoch in range(args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

            # Contrastive-loss weight is 0 for the first 5 epochs, then 5
            if epoch < 5:
                contrastive_weight = 0.0
            else:
                contrastive_weight = 5.0

            print(f"  Contrastive loss weight: {contrastive_weight}")

            avg_loss, avg_c_loss, avg_distill_loss, avg_avoid_loss = train_epoch(
                model, support_loader, memory_loader, optimizer, device,
                memory_features_dict, fake_anchor, real_anchor,
                20.0, args.distill_weight, contrastive_weight, args.temperature
            )

            print(f"  Train loss: {avg_loss:.4f} (contrastive: {avg_c_loss:.4f}, distill: {avg_distill_loss:.4f}, avoid: {avg_avoid_loss:.4f})")

            training_log.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'contrastive_loss': avg_c_loss,
                'distill_loss': avg_distill_loss,
                'avoid_loss': avg_avoid_loss
            })

        # Final validation
        final_results = validate_with_prototypes(
            model, validation_sets, fake_prototype, real_prototype,
            processor, device, args.xception_crop_size,
            batch_size=args.val_batch_size, num_workers=args.num_workers
        )

        # Define query_key and final_real_key (used for the final results)
        query_key = f'Query集({args.dataset_name})'
        final_real_key = 'final/real'

        print_validation_summary("Validation after training:", final_results)

        # Record the final average accuracy (used by the summary)
        final_query_acc = final_results.get(query_key, {}).get('total_acc', 0.0)
        final_real_acc = final_results.get(final_real_key, {}).get('total_acc', 0.0)
        best_avg_acc = (final_query_acc + final_real_acc) / 2.0

    # Save the few-shot model weights on demand (including the case where training is skipped)
    if args.save_fewshot_model:
        if isinstance(model, nn.DataParallel):
            fewshot_model_state = model.module.state_dict()
        else:
            fewshot_model_state = model.state_dict()
        fewshot_model_path = os.path.join(args.output_dir, 'fewshot_model.pth')
        torch.save({
            'model_state_dict': fewshot_model_state,
            'pretrain_checkpoint': args.pretrain_checkpoint,
            'dataset_name': args.dataset_name,
            'best_avg_acc': best_avg_acc,
            'config': vars(args),
        }, fewshot_model_path)
    else:
        print("[Few-shot] Model weights not saved (pass --save_fewshot_model to save)")

    # Save the results
    results_summary = {
        'dataset_name': args.dataset_name,
        'pretrain_checkpoint': args.pretrain_checkpoint,
        'initial_results': initial_results,
        'final_results': final_results,
        'training_log': training_log,
        'best_avg_acc': best_avg_acc,
        'config': vars(args)
    }

    results_path = os.path.join(args.output_dir, 'fewshot_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # Save the per-image confidence to separate files
    # Save the initial-result confidences
    initial_confidences_data = {}
    for dataset_name, result in initial_results.items():
        if 'confidences' in result and 'confidence_paths' in result:
            initial_confidences_data[dataset_name] = [
                {
                    'path': path,
                    'confidence': float(conf),
                    'label': int(label)
                }
                for path, conf, label in zip(
                    result['confidence_paths'],
                    result['confidences'],
                    validation_sets[dataset_name]['labels']
                )
            ]

    initial_confidences_path = os.path.join(args.output_dir, 'initial_confidences.json')
    with open(initial_confidences_path, 'w', encoding='utf-8') as f:
        json.dump(initial_confidences_data, f, indent=2, ensure_ascii=False)

    # Save the final-result confidences
    final_confidences_data = {}
    for dataset_name, result in final_results.items():
        if 'confidences' in result and 'confidence_paths' in result:
            final_confidences_data[dataset_name] = [
                {
                    'path': path,
                    'confidence': float(conf),
                    'label': int(label)
                }
                for path, conf, label in zip(
                    result['confidence_paths'],
                    result['confidences'],
                    validation_sets[dataset_name]['labels']
                )
            ]

    final_confidences_path = os.path.join(args.output_dir, 'final_confidences.json')
    with open(final_confidences_path, 'w', encoding='utf-8') as f:
        json.dump(final_confidences_data, f, indent=2, ensure_ascii=False)

    exp_elapsed = time.time() - exp_start_time
    print(
        f"[Few-shot] Done: {args.dataset_name}, elapsed {exp_elapsed:.1f} s, "
        f"output dir: {args.output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description='Few-shot Learning with Real Avoidance')

    # Path arguments (env vars also work; see README / .env.example)
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                       help='Pretrained checkpoint; env var FLEET_PRETRAIN_CHECKPOINT')
    parser.add_argument('--prototype_dir', type=str, default=None,
                       help='Prototype-vector directory; env var FLEET_PROTOTYPE_DIR')
    parser.add_argument('--query_dir', type=str, required=True,
                       help='Query dataset directory (new fake data)')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--dinov3_model_path', type=str, default=None,
                       help='DINOv3 local directory; if unset uses FLEET_DINOV3_MODEL_PATH, then <repo>/weights/dinov3-vitl16-pretrain-lvd1689m')

    # Model arguments
    parser.add_argument('--xception_feature_dim', type=int, default=1024)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--xception_crop_size', type=int, default=128)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--distill_weight', type=float, default=10.0)
    parser.add_argument('--force_train', action='store_true', help='Force training even if the pre-training Query-set accuracy is > 90')
    parser.add_argument('--save_fewshot_model', action='store_true', help='Save the few-shot fine-tuned model weights (off by default)')

    # Data-path arguments (for final/fake and final/real)
    parser.add_argument('--real_val_dir', type=str, default=None,
                        help='Real validation directory; env var FLEET_REAL_VAL_DIR')
    parser.add_argument('--aigibench_train_dir', type=str, default=None,
                        help='AIGIBench training set (Memory set); env var FLEET_AIGIBENCH_TRAIN')
    parser.add_argument('--aigibench_val_dir', type=str, default=None,
                        help='AIGIBench validation set; env var FLEET_AIGIBENCH_VAL')

    # Support-set settings
    parser.add_argument('--n_fake_support', type=int, default=10)
    parser.add_argument('--n_real_support', type=int, default=10)
    parser.add_argument('--max_real_val', type=int, default=1000, help='Max number of final/real validation images to sample (None = unlimited)')

    # Memory set (replay set) settings
    parser.add_argument('--n_fake_memory', type=int, default=500, help='Number of fake images in the Memory set')
    parser.add_argument('--n_real_memory', type=int, default=500, help='Number of real images in the Memory set')

    # Performance arguments
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader image-loading threads')
    parser.add_argument('--val_batch_size', type=int, default=256, help='Validation batch size (can be larger for multi-GPU)')

    args = parser.parse_args()

    args.pretrain_checkpoint = coalesce_cli_env(args.pretrain_checkpoint, "FLEET_PRETRAIN_CHECKPOINT")
    args.prototype_dir = coalesce_cli_env(args.prototype_dir, "FLEET_PROTOTYPE_DIR")
    raw_dino = coalesce_cli_env(args.dinov3_model_path, "FLEET_DINOV3_MODEL_PATH")
    args.dinov3_model_path = resolve_dinov3_model_path(raw_dino if raw_dino else None)
    args.real_val_dir = coalesce_cli_env(args.real_val_dir, "FLEET_REAL_VAL_DIR")
    args.aigibench_train_dir = coalesce_cli_env(args.aigibench_train_dir, "FLEET_AIGIBENCH_TRAIN")
    args.aigibench_val_dir = coalesce_cli_env(args.aigibench_val_dir, "FLEET_AIGIBENCH_VAL")

    missing = []
    if not args.pretrain_checkpoint:
        missing.append("(--pretrain_checkpoint or FLEET_PRETRAIN_CHECKPOINT)")
    if not args.prototype_dir:
        missing.append("(--prototype_dir or FLEET_PROTOTYPE_DIR)")
    if not args.real_val_dir:
        missing.append("(--real_val_dir or FLEET_REAL_VAL_DIR)")
    if not args.aigibench_train_dir:
        missing.append("(--aigibench_train_dir or FLEET_AIGIBENCH_TRAIN)")
    if not args.aigibench_val_dir:
        missing.append("(--aigibench_val_dir or FLEET_AIGIBENCH_VAL)")
    if missing:
        parser.error("Missing required paths: " + "; ".join(missing))
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the experiment
    run_experiment(args)


if __name__ == '__main__':
    main()
