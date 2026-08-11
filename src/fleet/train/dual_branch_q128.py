"""Contrastive pretraining with DINOv3 + Xception dual branches (with attention-weight orthogonal loss and presence loss; no residual connection; Q dim = 128).

Set ``CUDA_VISIBLE_DEVICES`` before running. Data and weight paths are configured via environment variables; see the repo ``.env.example``.
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
import numpy as np
import glob
import random
import json

from fleet.models.dual_branch_contrastive import DualBranchContrastiveModel
from fleet.utils import (
    resolve_dinov3_model_path,
    apply_fft_high_freq_mask,
    pretrain_contrastive_loss as contrastive_loss,
    attention_orthogonal_loss,
    attention_coverage_loss,
)
from fleet.datasets import AIGIBenchDataset, ImageDatasetForValidation, collect_aigibench_image_paths


def _open_progress_stream():
    """Open a stream to the controlling terminal; disable progress (return None) when there is none, so progress is never written to redirected logs."""
    try:
        return open('/dev/tty', 'w', encoding='utf-8', buffering=1)
    except OSError:
        return None


_PROGRESS_STREAM = _open_progress_stream()


def console_tqdm(iterable=None, **kwargs):
    kwargs.setdefault('disable', _PROGRESS_STREAM is None)
    if _PROGRESS_STREAM is not None:
        kwargs.setdefault('file', _PROGRESS_STREAM)
        kwargs.setdefault('dynamic_ncols', True)
    kwargs.setdefault('mininterval', 1.0)
    return tqdm(iterable, **kwargs)


def train_contrastive_epoch(model, dataloader, optimizer, device, temperature=0.07,
                           attn_orth_weight=0.1, attn_cov_weight=0.1):
    """Train one contrastive-learning epoch (with attention-weight orthogonal loss and presence loss)."""
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_attn_orth_loss = 0.0
    total_attn_cov_loss = 0.0
    total_samples = 0
    
    pbar = console_tqdm(dataloader, desc="Contrastive training")
    for images_dinov3, images_xception, labels in pbar:
        images_dinov3 = images_dinov3.to(device)
        images_xception = images_xception.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        features, attn_weights = model(images_dinov3, images_xception)
        contrastive_loss_val = contrastive_loss(features, labels, temperature=temperature)
        attn_orth_loss_val = attention_orthogonal_loss(attn_weights, labels)
        attn_cov_loss_val = attention_coverage_loss(attn_weights, labels)
        total_loss_val = contrastive_loss_val + attn_orth_weight * attn_orth_loss_val + attn_cov_weight * attn_cov_loss_val
        total_loss_val.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += total_loss_val.item() * batch_size
        total_contrastive_loss += contrastive_loss_val.item() * batch_size
        total_attn_orth_loss += attn_orth_loss_val.item() * batch_size
        total_attn_cov_loss += attn_cov_loss_val.item() * batch_size
        total_samples += batch_size
        current_loss = total_loss / total_samples if total_samples > 0 else 0
        current_contrastive = total_contrastive_loss / total_samples if total_samples > 0 else 0
        current_attn_orth = total_attn_orth_loss / total_samples if total_samples > 0 else 0
        current_attn_cov = total_attn_cov_loss / total_samples if total_samples > 0 else 0
        current_attn_orth_weighted = current_attn_orth * attn_orth_weight
        current_attn_cov_weighted = current_attn_cov * attn_cov_weight
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'c_loss': f'{current_contrastive:.4f}',
            'orth': f'{current_attn_orth:.4f}->{current_attn_orth_weighted:.4f}',
            'cov': f'{current_attn_cov:.4f}->{current_attn_cov_weighted:.4f}'
        })
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_contrastive = total_contrastive_loss / total_samples if total_samples > 0 else 0
    avg_attn_orth = total_attn_orth_loss / total_samples if total_samples > 0 else 0
    avg_attn_cov = total_attn_cov_loss / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_contrastive, avg_attn_orth, avg_attn_cov


def validate_contrastive(model, dataloader, device, temperature=0.07, 
                        attn_orth_weight=0.1, attn_cov_weight=0.1):
    """Contrastive-learning validation (with attention-weight orthogonal loss and presence loss)."""
    model.eval()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_attn_orth_loss = 0.0
    total_attn_cov_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images_dinov3, images_xception, labels in console_tqdm(dataloader, desc="Validating"):
            images_dinov3 = images_dinov3.to(device)
            images_xception = images_xception.to(device)
            labels = labels.to(device)
            
            features, attn_weights = model(images_dinov3, images_xception)
            
            contrastive_loss_val = contrastive_loss(features, labels, temperature=temperature)
            attn_orth_loss_val = attention_orthogonal_loss(attn_weights, labels)
            attn_cov_loss_val = attention_coverage_loss(attn_weights, labels)
            total_loss_val = contrastive_loss_val + attn_orth_weight * attn_orth_loss_val + attn_cov_weight * attn_cov_loss_val
            
            batch_size = labels.size(0)
            total_loss += total_loss_val.item() * batch_size
            total_contrastive_loss += contrastive_loss_val.item() * batch_size
            total_attn_orth_loss += attn_orth_loss_val.item() * batch_size
            total_attn_cov_loss += attn_cov_loss_val.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_contrastive = total_contrastive_loss / total_samples if total_samples > 0 else 0
    avg_attn_orth = total_attn_orth_loss / total_samples if total_samples > 0 else 0
    avg_attn_cov = total_attn_cov_loss / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_contrastive, avg_attn_orth, avg_attn_cov


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dual-branch contrastive pretraining")
    # Paths
    parser.add_argument('--dinov3_model_path', type=str, default=None, help='DINOv3 weights path; auto-resolved by default')
    parser.add_argument('--aigibench_train', type=str, required=True, help='AIGIBench training set directory')
    parser.add_argument('--aigibench_val', type=str, required=True, help='AIGIBench validation set directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/checkpoints_q128', help='Output directory')
    # Model hyperparameters (commonly tuned)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--feature_layer', type=int, default=20)
    # Training hyperparameters (commonly tuned)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--eval_every_steps', type=int, default=300)
    parser.add_argument('--target_acc', type=float, default=98.0, help='AIGIBench/val early-stop threshold')
    parser.add_argument('--max_train_steps', type=int, default=2000, help='0 = unlimited')
    parser.add_argument('--force_train', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    dinov3_model_path = args.dinov3_model_path or resolve_dinov3_model_path()
    print(f"DINOv3 path (resolved): {dinov3_model_path}")
    aigibench_train_dir = args.aigibench_train
    aigibench_val_dir = args.aigibench_val
    output_dir = args.output_dir

    # Contrastive pretraining parameters
    xception_feature_dim = 1024
    projection_dim = args.projection_dim
    feature_layer = args.feature_layer
    use_last_hidden_state = False  # hardcoded: do not use last_hidden_state
    normalize_feature = True  # hardcoded: apply the final LayerNorm to hidden_states[feature_layer]
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.0
    num_heads = 8
    temperature = 0.07
    learning_rate = 1e-4  # hardcoded
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    eval_every_steps = args.eval_every_steps
    target_acc = args.target_acc
    max_train_steps = args.max_train_steps
    force_train = bool(args.force_train)
    attn_orth_weight = 0.1  # attention-weight orthogonal loss weight (repulsion loss)
    attn_cov_weight = 0.0001   # attention-weight presence loss weight (coverage loss)
    
    # GPU setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU, PyTorch visible devices: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check whether a trained checkpoint already exists
    pretrain_checkpoint_path = os.path.join(output_dir, 'pretrain_dual_branch_with_attn_loss_and_coverage_no_residual_q128.pth')
    skip_training = os.path.exists(pretrain_checkpoint_path) and not force_train
    
    if skip_training:
        print(f"[Pretrain] Existing checkpoint detected, skipping training and validating: {pretrain_checkpoint_path}")
    else:
        step_limit_desc = "unlimited" if max_train_steps <= 0 else f"{max_train_steps} step"
        print(
            f"[Pretrain] Start: dim={projection_dim}, "
            f"orth={attn_orth_weight}, coverage={attn_cov_weight}, "
            f"eval_interval={eval_every_steps}, max_steps={step_limit_desc}"
        )
    
    # Image preprocessing (DINOv3 default preprocessing)
    print("Loading DINOv3 image processor...")
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(
            dinov3_model_path,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: failed to load DINOv3 processor: {e}")
        print("Falling back to default ImageNet preprocessing...")
        processor = None
    
    # Load AIGIBench training set (needed for validation)
    print(f"\nLoading AIGIBench training set from: {aigibench_train_dir}")
    train_dataset = AIGIBenchDataset(aigibench_train_dir, processor=processor, xception_crop_size=128)
    
    if len(train_dataset) == 0:
        print("Warning: AIGIBench training set is empty, exiting")
        return
    
    # Create the contrastive-learning model
    model = DualBranchContrastiveModel(
        dinov3_model_path=dinov3_model_path,
        xception_feature_dim=xception_feature_dim,
        projection_dim=projection_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_heads=num_heads,
        q_dim=128,
        feature_layer=feature_layer,
        use_last_hidden_state=use_last_hidden_state,
        normalize_feature=normalize_feature,
    )
    model = model.to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # If a checkpoint already exists, load it and skip training
    if skip_training:
        print(f"\nLoading trained model: {pretrain_checkpoint_path}")
        checkpoint = torch.load(pretrain_checkpoint_path, map_location=device, weights_only=False)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded")
    else:
        # Training mode: load the validation set and create data loaders
        print(f"\nLoading AIGIBench validation set from: {aigibench_val_dir}")
        val_dataset = AIGIBenchDataset(aigibench_val_dir, processor=processor, xception_crop_size=128)
        
        # Create data loaders
        num_workers = min(8, os.cpu_count() or 1)
        print(f"Using {num_workers} worker threads for data loading")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        ) if len(val_dataset) > 0 else None
        
        # Create the optimizer (train LoRA params, Xception params, routing-fusion params, projection params)
        if isinstance(model, nn.DataParallel):
            trainable_params = []
            # LoRA params
            for name, lora_layer in model.module.lora_modules:
                trainable_params.append(lora_layer.lora_A)
                trainable_params.append(lora_layer.lora_B)
            # Xception params
            trainable_params.extend(model.module.xception.parameters())
            trainable_params.extend(model.module.xception_fc.parameters())
            # Routing-fusion params
            trainable_params.extend(model.module.routing.parameters())
            # Projection params
            trainable_params.extend(model.module.projection.parameters())
            # DINOv3 projection params (if present)
            if hasattr(model.module, 'dinov3_proj') and not isinstance(model.module.dinov3_proj, nn.Identity):
                trainable_params.extend(model.module.dinov3_proj.parameters())
        else:
            trainable_params = []
            # LoRA params
            for name, lora_layer in model.lora_modules:
                trainable_params.append(lora_layer.lora_A)
                trainable_params.append(lora_layer.lora_B)
            # Xception params
            trainable_params.extend(model.xception.parameters())
            trainable_params.extend(model.xception_fc.parameters())
            # Routing-fusion params
            trainable_params.extend(model.routing.parameters())
            # Projection params
            trainable_params.extend(model.projection.parameters())
            # DINOv3 projection params (if present)
            if hasattr(model, 'dinov3_proj') and not isinstance(model.dinov3_proj, nn.Identity):
                trainable_params.extend(model.dinov3_proj.parameters())
        
        # Keep only params with requires_grad=True to avoid passing frozen params to the optimizer
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Check trainable params
        trainable_params_count = sum(p.numel() for p in trainable_params)
        total_params_count = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable params: {trainable_params_count:,} / total: {total_params_count:,}")
        print(f"Param efficiency: {100 * trainable_params_count / total_params_count:.2f}%")
        
        print("Start dual-branch contrastive pretraining")

        best_val_acc = -1.0
        global_step = 0
        last_train_loss = 0.0
        last_train_contrastive = 0.0
        last_train_attn_orth = 0.0
        last_train_attn_cov = 0.0

        prototype_dir = os.path.join(output_dir, 'prototypes_dual_branch_with_attn_loss_and_coverage_no_residual_q128_freq')

        def save_checkpoint(epoch_idx, step, train_loss, val_result, fake_prototype, real_prototype, reached_target, validation_results=None):
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            ckpt_payload = {
                'epoch': epoch_idx + 1,
                'global_step': step,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_result.get('total_acc', 0.0),
                'val_fake_acc': val_result.get('fake_acc', 0.0),
                'val_real_acc': val_result.get('real_acc', 0.0),
                'reached_target': reached_target,
                'target_acc': target_acc,
                'projection_dim': projection_dim,
                'feature_layer': feature_layer,
                'use_last_hidden_state': use_last_hidden_state,
                'normalize_feature': normalize_feature,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'num_heads': num_heads,
                'q_dim': 128,
                'attn_orth_weight': attn_orth_weight,
                'attn_cov_weight': attn_cov_weight,
                'eval_every_steps': eval_every_steps,
                'max_train_steps': max_train_steps,
                'validation_results': validation_results or {},
            }

            step_ckpt_dir = os.path.join(output_dir, 'checkpoints_by_step')
            os.makedirs(step_ckpt_dir, exist_ok=True)
            step_ckpt_path = os.path.join(step_ckpt_dir, f'pretrain_step{step:06d}.pth')
            torch.save(ckpt_payload, step_ckpt_path)
            torch.save(ckpt_payload, pretrain_checkpoint_path)

            step_proto_dir = os.path.join(output_dir, 'prototypes_by_step', f'step{step:06d}')
            os.makedirs(step_proto_dir, exist_ok=True)
            np.save(os.path.join(step_proto_dir, 'fake_prototype.npy'), fake_prototype)
            np.save(os.path.join(step_proto_dir, 'real_prototype.npy'), real_prototype)
            os.makedirs(prototype_dir, exist_ok=True)
            np.save(os.path.join(prototype_dir, 'fake_prototype.npy'), fake_prototype)
            np.save(os.path.join(prototype_dir, 'real_prototype.npy'), real_prototype)

            result_path = None
            if validation_results is not None:
                step_result_dir = os.path.join(output_dir, 'validation_by_step')
                os.makedirs(step_result_dir, exist_ok=True)
                result_path = os.path.join(step_result_dir, f'validation_step{step:06d}.json')
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(validation_results, f, indent=2, ensure_ascii=False)

            saved_items = f"checkpoint={step_ckpt_path}, prototypes={step_proto_dir}"
            if result_path:
                saved_items += f", validation={result_path}"
            print(f"Step {step} saved: {saved_items}")

        stop_training = False
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            model.train()
            epoch_loss_sum = 0.0
            epoch_contrastive_sum = 0.0
            epoch_attn_orth_sum = 0.0
            epoch_attn_cov_sum = 0.0
            epoch_samples = 0

            pbar = console_tqdm(train_loader, desc="Contrastive training")
            for images_dinov3, images_xception, labels in pbar:
                images_dinov3 = images_dinov3.to(device)
                images_xception = images_xception.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                features, attn_weights = model(images_dinov3, images_xception)
                contrastive_loss_val = contrastive_loss(features, labels, temperature=temperature)
                attn_orth_loss_val = attention_orthogonal_loss(attn_weights, labels)
                attn_cov_loss_val = attention_coverage_loss(attn_weights, labels)
                total_loss_val = contrastive_loss_val + attn_orth_weight * attn_orth_loss_val + attn_cov_weight * attn_cov_loss_val
                total_loss_val.backward()
                optimizer.step()

                global_step += 1
                batch_size_now = labels.size(0)
                epoch_loss_sum += total_loss_val.item() * batch_size_now
                epoch_contrastive_sum += contrastive_loss_val.item() * batch_size_now
                epoch_attn_orth_sum += attn_orth_loss_val.item() * batch_size_now
                epoch_attn_cov_sum += attn_cov_loss_val.item() * batch_size_now
                epoch_samples += batch_size_now

                last_train_loss = epoch_loss_sum / epoch_samples
                last_train_contrastive = epoch_contrastive_sum / epoch_samples
                last_train_attn_orth = epoch_attn_orth_sum / epoch_samples
                last_train_attn_cov = epoch_attn_cov_sum / epoch_samples
                pbar.set_postfix({
                    'step': global_step,
                    'loss': f'{last_train_loss:.4f}',
                    'c_loss': f'{last_train_contrastive:.4f}',
                    'orth': f'{last_train_attn_orth:.4f}',
                    'cov': f'{last_train_attn_cov:.4f}',
                })

                if eval_every_steps > 0 and global_step % eval_every_steps == 0:
                    print(f"\n--- Step {global_step} evaluate & save ---")
                    model.eval()
                    print("Extracting training-set prototypes...")
                    fake_prototype, real_prototype = extract_prototypes_from_train_set(
                        model, train_dataset, device, processor, xception_crop_size=128
                    )
                    validation_results = validate_on_datasets(
                        model, fake_prototype, real_prototype, aigibench_val_dir, device, processor, xception_crop_size=128
                    )
                    val_result = validation_results.get('AIGIBench/val', {'total_acc': 0.0, 'fake_acc': 0.0, 'real_acc': 0.0})
                    val_acc = val_result.get('total_acc', 0.0)
                    reached_target = val_acc >= target_acc
                    best_val_acc = max(best_val_acc, val_acc)
                    print(f"Step {global_step} AIGIBench/val={val_acc:.2f}% (target={target_acc:.2f}%, reached={reached_target}), saving checkpoint")
                    save_checkpoint(
                        epoch, global_step, last_train_loss, val_result,
                        fake_prototype, real_prototype, reached_target,
                        validation_results=validation_results
                    )
                    model.train()
                    if reached_target:
                        print(f"Step {global_step} AIGIBench/val={val_acc:.2f}% reached the target {target_acc:.2f}%, early stop")
                        stop_training = True
                        break
                    if max_train_steps > 0 and global_step >= max_train_steps:
                        print(f"Reached the max training steps {max_train_steps}, stopping training")
                        stop_training = True
                        break

            print(f"Train loss: {last_train_loss:.4f} (contrastive: {last_train_contrastive:.4f}, orth: {last_train_attn_orth:.4f}, cov: {last_train_attn_cov:.4f})")
            if stop_training:
                break
        if not os.path.exists(pretrain_checkpoint_path):
            print("\nNo eval/save was triggered during training, performing a final fallback save...")
            model.eval()
            fake_prototype, real_prototype = extract_prototypes_from_train_set(
                model, train_dataset, device, processor, xception_crop_size=128
            )
            val_result = validate_single_dataset(
                model, fake_prototype, real_prototype, aigibench_val_dir, device, processor,
                dataset_type='aigibench', xception_crop_size=128
            )
            best_val_acc = val_result.get('total_acc', 0.0)
            save_checkpoint(num_epochs - 1, global_step, last_train_loss, val_result, fake_prototype, real_prototype, False)

        print(f"[Pretrain] Training finished, best validation accuracy: {best_val_acc:.2f}%")
    
    # After training (or with an existing checkpoint), load the best weights and corresponding prototypes, then validate
    print("Loading best weights and corresponding prototypes for validation")
    
    # If training just finished, load the best weights
    if not skip_training:
        print("\nLoading best training weights...")
        if os.path.exists(pretrain_checkpoint_path):
            checkpoint = torch.load(pretrain_checkpoint_path, map_location=device, weights_only=False)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best weights loaded (Epoch {checkpoint.get('epoch', '?')}, Step {checkpoint.get('global_step', '?')}, val accuracy: {checkpoint.get('val_acc', checkpoint.get('combined_acc', 0.0)):.2f}%)")
        else:
            print("Warning: training checkpoint not found, using the current model")
    
    # Check whether the prototypes for the best weights already exist
    prototype_dir = os.path.join(output_dir, 'prototypes_dual_branch_with_attn_loss_and_coverage_no_residual_q128_freq')
    fake_prototype_path = os.path.join(prototype_dir, 'fake_prototype.npy')
    real_prototype_path = os.path.join(prototype_dir, 'real_prototype.npy')
    
    if os.path.exists(fake_prototype_path) and os.path.exists(real_prototype_path):
        print("\nLoading prototypes for the best weights...")
        fake_prototype = np.load(fake_prototype_path)
        real_prototype = np.load(real_prototype_path)
        print(f"Prototypes loaded: Fake={fake_prototype.shape}, Real={real_prototype.shape}")
    else:
        # If no prototypes are found (backward-compat), re-extract them with the current model
        print("\nPrototypes for the best weights not found, re-extracting with the current model...")
        fake_prototype, real_prototype = extract_prototypes_from_train_set(
            model, train_dataset, device, processor, xception_crop_size=128
        )
        
        # Save the prototypes
        os.makedirs(prototype_dir, exist_ok=True)
        np.save(fake_prototype_path, fake_prototype)
        np.save(real_prototype_path, real_prototype)
        print(f"Prototypes saved to: {prototype_dir}")
    
    # Validate on the validation set
    print("\nValidating on the validation set...")
    validation_results = validate_on_datasets(
        model, fake_prototype, real_prototype, aigibench_val_dir, device, processor, xception_crop_size=128
    )
    
    # Save the validation results
    results_path = os.path.join(output_dir, 'validation_results_dual_branch_with_attn_loss_and_coverage_no_residual_q128_freq.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    print(f"Validation results (saved to {results_path}):")
    for dataset_name, result in validation_results.items():
        print(
            f"  {dataset_name}: total={result['total_acc']:.2f}% "
            f"fake={result['fake_acc']:.2f}% real={result['real_acc']:.2f}%"
        )


def extract_prototypes_from_train_set(model, train_dataset, device, processor, xception_crop_size=128, batch_size=128):
    """Extract prototypes from the training set (dual-branch version)."""
    model.eval()
    
    # Separate fake and real samples
    fake_paths = []
    real_paths = []
    for idx in range(len(train_dataset)):
        label = train_dataset.labels[idx]
        img_path = train_dataset.image_paths[idx]
        if label == 0:  # fake
            fake_paths.append(img_path)
        else:  # real
            real_paths.append(img_path)
    
    original_fake_count = len(fake_paths)
    original_real_count = len(real_paths)
    print(f"  Extracting prototypes: fake={original_fake_count}, real={original_real_count}")
    
    # If more than 50000 images, randomly sample 50000
    max_samples = 50000
    if len(fake_paths) > max_samples:
        random.seed(42)  # fixed seed for reproducibility
        fake_paths = random.sample(fake_paths, max_samples)
        print(f"  Fake sampled to: {len(fake_paths)} (from {original_fake_count})")
    
    if len(real_paths) > max_samples:
        random.seed(43)  # different seed so fake and real sampling are independent
        real_paths = random.sample(real_paths, max_samples)
        print(f"  Real sampled to: {len(real_paths)} (from {original_real_count})")
    
    # Extract fake and real features
    fake_features = extract_features_batch(model, fake_paths, device, processor, xception_crop_size, batch_size)
    real_features = extract_features_batch(model, real_paths, device, processor, xception_crop_size, batch_size)
    
    # Compute prototypes (mean features)
    fake_prototype = np.mean(fake_features, axis=0)
    real_prototype = np.mean(real_features, axis=0)
    print(f"  Prototypes extracted: fake={fake_prototype.shape}, real={real_prototype.shape}")
    
    return fake_prototype, real_prototype


def extract_features_batch(model, image_paths, device, processor, xception_crop_size=128, batch_size=128):
    """Extract features in batches (dual-branch version)."""
    dataset = ImageDatasetForValidation(image_paths, processor, xception_crop_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    all_features = []
    with torch.no_grad():
        for pixel_values_dinov3, pixel_values_xception in console_tqdm(dataloader, desc="Extracting features", leave=False):
            pixel_values_dinov3 = pixel_values_dinov3.to(device)
            pixel_values_xception = pixel_values_xception.to(device)
            features, _ = model(pixel_values_dinov3, pixel_values_xception)  # ignore attention weights
            all_features.append(features.cpu().numpy())
    
    return np.vstack(all_features)


def validate_on_datasets(model, fake_prototype, real_prototype, aigibench_val_dir, device, processor, xception_crop_size=128):
    """Validate: only AIGIBench/val (early-stop criterion). Full fake/real evaluation is done by a separate script."""
    results = {}

    if aigibench_val_dir and os.path.exists(aigibench_val_dir):
        print("Validating AIGIBench/val...")
        result = validate_single_dataset(
            model, fake_prototype, real_prototype, aigibench_val_dir, device, processor,
            dataset_type='aigibench', xception_crop_size=xception_crop_size
        )
        results['AIGIBench/val'] = result
        print(f"  AIGIBench/val: total={result['total_acc']:.2f}% fake={result['fake_acc']:.2f}% real={result['real_acc']:.2f}% (n={result['n_samples']})")

    return results


def validate_single_dataset(model, fake_prototype, real_prototype, data_dir, device, processor, 
                           dataset_type='aigibench', max_samples=None, xception_crop_size=128):
    """Validate on a single dataset (dual-branch version)."""
    # Collect image paths and labels
    image_paths = []
    labels = []
    
    if dataset_type == 'aigibench':
        image_paths, labels = collect_aigibench_image_paths(data_dir)
    
    elif dataset_type == 'fewshot_fake':
        # fewshot-datast layout: directly contains fake images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
            image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext.upper()), recursive=True))
        labels = [0] * len(image_paths)  # all fake
    
    elif dataset_type == 'cc12m_real':
        # cc12m-2mp-realistic layout: directly contains real images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
            image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext.upper()), recursive=True))
        labels = [1] * len(image_paths)  # all real
    
    if len(image_paths) == 0:
        print(f"  Warning: dataset is empty")
        return {
            'total_acc': 0.0,
            'fake_acc': 0.0,
            'real_acc': 0.0,
            'n_samples': 0
        }
    
    # If max_samples is specified, randomly sample
    if max_samples is not None and len(image_paths) > max_samples:
        random.seed(42)  # fixed seed for reproducibility
        total_count = len(image_paths)
        indices = random.sample(range(total_count), max_samples)
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(f"  Randomly sampled {max_samples} from {total_count} images")
    
    print(f"  Found {len(image_paths)} images (Fake: {sum(1 for l in labels if l == 0)}, Real: {sum(1 for l in labels if l == 1)})")
    
    # Extract features
    features = extract_features_batch(model, image_paths, device, processor, xception_crop_size)
    
    # Predict using the prototypes
    fake_sim = np.dot(features, fake_prototype) / (
        np.linalg.norm(features, axis=1) * np.linalg.norm(fake_prototype)
    )
    real_sim = np.dot(features, real_prototype) / (
        np.linalg.norm(features, axis=1) * np.linalg.norm(real_prototype)
    )
    
    # Prediction: real_sim > fake_sim -> real (1), otherwise -> fake (0)
    predictions = (real_sim > fake_sim).astype(int)
    
    # Compute accuracy
    labels_array = np.array(labels)
    total_acc = (predictions == labels_array).mean() * 100
    
    # Compute fake and real accuracy separately
    fake_mask = labels_array == 0
    real_mask = labels_array == 1
    
    fake_acc = 0.0
    real_acc = 0.0
    if fake_mask.sum() > 0:
        fake_acc = (predictions[fake_mask] == labels_array[fake_mask]).mean() * 100
    if real_mask.sum() > 0:
        real_acc = (predictions[real_mask] == labels_array[real_mask]).mean() * 100
    
    return {
        'total_acc': float(total_acc),
        'fake_acc': float(fake_acc),
        'real_acc': float(real_acc),
        'n_samples': len(image_paths)
    }


if __name__ == "__main__":
    main()
