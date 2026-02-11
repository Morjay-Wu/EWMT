import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import rasterio

from data_cia import PatchSet, Mode
from model_EWMT import FusionNet_2, CompoundLoss, Pretrained


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "checkpoints"
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    val_split: float = 0.1
    patch_size: int = 128
    patch_stride: Optional[int] = None
    patch_padding: int = 0
    seed: int = 42
    amp: bool = True
    resume: Optional[str] = None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def detect_image_size(data_dir: Path) -> Tuple[int, int]:
    for scene_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        tif_candidates = sorted(scene_dir.glob("*.tif"))
        for tif in tif_candidates:
            try:
                with rasterio.open(str(tif)) as ds:
                    return int(ds.height), int(ds.width)
            except Exception:
                continue
    raise RuntimeError(f"Failed to detect image size in {data_dir}")


def collate_patches_to_batch(batch: List[List[torch.Tensor]]):
    valid_samples = [sample for sample in batch if isinstance(sample, list) and len(sample) >= 4]
    if len(valid_samples) == 0:
        # Fallback: keep dataloader running; create an empty batch that will be skipped
        return [torch.empty(0)], torch.empty(0)

    modis_ref = torch.stack([s[0] for s in valid_samples], dim=0)
    land_ref = torch.stack([s[1] for s in valid_samples], dim=0)
    modis_pre = torch.stack([s[2] for s in valid_samples], dim=0)
    land_pre = torch.stack([s[3] for s in valid_samples], dim=0)

    return [modis_ref, land_ref, modis_pre], land_pre


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    with torch.no_grad():
        pred = torch.clamp(pred, 0.0, max_val)
        target = torch.clamp(target, 0.0, max_val)
        mse = torch.mean((pred - target) ** 2).item()
        if mse == 0:
            return 100.0
        return float(10.0 * np.log10((max_val ** 2) / mse))


def _build_loader_for_dir(dir_path: Path, cfg: TrainConfig, mode: Mode) -> Tuple[DataLoader, Tuple[int, int], int]:
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    image_size = detect_image_size(dir_path)
    patch_stride = cfg.patch_stride if cfg.patch_stride is not None else cfg.patch_size

    dataset: Dataset = PatchSet(
        image_dir=dir_path,
        image_size=image_size,
        patch_size=(cfg.patch_size, cfg.patch_size),
        patch_stride=(patch_stride, patch_stride),
        patch_padding=(cfg.patch_padding, cfg.patch_padding),
        mode=mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(mode == Mode.TRAINING),
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_patches_to_batch,
        drop_last=False,
    )
    return loader, image_size, len(dataset)


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, Tuple[int, int], Tuple[int, int]]:
    root = Path(cfg.data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root directory not found: {root}")

    train_dir = root / "train"
    val_dir = root / "test"

    train_loader, train_image_size, train_len = _build_loader_for_dir(train_dir, cfg, Mode.TRAINING)
    val_loader, val_image_size, val_len = _build_loader_for_dir(val_dir, cfg, Mode.VALIDATION)

    return train_loader, val_loader, train_image_size, val_image_size


def save_checkpoint(state: dict, out_dir: Path, is_best: bool, epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pth"
    torch.save(state, str(last_path))
    if is_best:
        best_path = out_dir / "best.pth"
        torch.save(state, str(best_path))


def maybe_resume(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: Optional[GradScaler], scheduler, resume_path: Optional[str]) -> Tuple[int, float]:
    if resume_path is None:
        return 0, float("inf")
    ckpt_path = Path(resume_path)
    if not ckpt_path.is_file():
        print(f"[warn] Resume path not found: {ckpt_path}")
        return 0, float("inf")
    ckpt = torch.load(str(ckpt_path), map_location="cuda")
    model.load_state_dict(ckpt["model"])  # type: ignore[index]
    optimizer.load_state_dict(ckpt["optimizer"])  # type: ignore[index]
    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])  # type: ignore[index]
        except Exception:
            pass
    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])  # type: ignore[index]
        except Exception:
            pass
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    print(f"[info] Resumed from {resume_path} at epoch {start_epoch-1}")
    return start_epoch, best_val


def train_one_epoch(model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader, device: torch.device, scaler: Optional[GradScaler], use_amp: bool) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Train", ncols=100)
    for maybe_inputs, maybe_target in pbar:
        # Skip empty batches from collate fallback
        if isinstance(maybe_inputs, list) and len(maybe_inputs) == 1 and maybe_inputs[0].numel() == 0:
            continue

        inputs: List[torch.Tensor] = [t.to(device, non_blocking=True) for t in maybe_inputs]
        target: torch.Tensor = maybe_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                output = model(inputs)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        running_psnr += compute_psnr(output.detach(), target.detach(), max_val=1.0)
        num_batches += 1
        pbar.set_postfix({"loss": f"{running_loss/num_batches:.4f}", "psnr": f"{running_psnr/num_batches:.2f}"})

    mean_loss = running_loss / max(1, num_batches)
    mean_psnr = running_psnr / max(1, num_batches)
    return mean_loss, mean_psnr


@torch.no_grad()
def validate(model: nn.Module, criterion: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Valid", ncols=100)
    for maybe_inputs, maybe_target in pbar:
        if isinstance(maybe_inputs, list) and len(maybe_inputs) == 1 and maybe_inputs[0].numel() == 0:
            continue

        inputs: List[torch.Tensor] = [t.to(device, non_blocking=True) for t in maybe_inputs]
        target: torch.Tensor = maybe_target.to(device, non_blocking=True)

        output = model(inputs)
        loss = criterion(output, target)

        total_loss += float(loss.item())
        total_psnr += compute_psnr(output, target, max_val=1.0)
        num_batches += 1
        pbar.set_postfix({"loss": f"{total_loss/num_batches:.4f}", "psnr": f"{total_psnr/num_batches:.2f}"})

    return total_loss / max(1, num_batches), total_psnr / max(1, num_batches)


def build_model_and_loss(device: torch.device) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, object]:
    # Require CUDA due to internal module device assumptions
    if device.type != "cuda":
        raise RuntimeError("FusionNet_2 currently requires CUDA. Please run on a CUDA-enabled device.")

    model = FusionNet_2().to(device)

    pretrained = Pretrained().to(device)
    pretrained.eval()
    for p in pretrained.parameters():
        p.requires_grad = False

    criterion = CompoundLoss(pretrained=pretrained, alpha=0.85, normalize=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    return model, criterion, optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Train FusionNet_2 on CIA dataset patches")
    parser.add_argument("--data_dir", type=str, default=r"D:\研究生\实验室数据集\时空融合\CIA1", help="Root directory containing scene subfolders")
    parser.add_argument("--out_dir", type=str, default=r"D:\研究生\论文编写\EWMT改良\out_cia\wtq9", help="Directory to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--patch_stride", type=int, default=None)
    parser.add_argument("--patch_padding", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Assemble config
    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_split=args.val_split,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_padding=args.patch_padding,
        seed=args.seed,
        amp=not args.no_amp,
        resume=args.resume,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(Path(cfg.out_dir) / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    set_seed(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this training script due to model internals.")
    device = torch.device("cuda")

    train_loader, val_loader, train_image_size, val_image_size = build_dataloaders(cfg)

    model, criterion, optimizer, scheduler = build_model_and_loss(device)

    # Override optimizer hyperparams based on CLI
    for g in optimizer.param_groups:
        g["lr"] = cfg.lr
        g["weight_decay"] = cfg.weight_decay

    scaler = GradScaler(enabled=cfg.amp)

    start_epoch, best_val = maybe_resume(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        resume_path=cfg.resume,
    )

    print(f"Train image size: {train_image_size}, Val image size: {val_image_size}")
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    print(f"Using AMP: {cfg.amp}")

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")

        train_loss, train_psnr = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            scaler=scaler,
            use_amp=cfg.amp,
        )

        val_loss, val_psnr = validate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
        )

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        print(f"Train   - loss: {train_loss:.4f}, psnr: {train_psnr:.2f}")
        print(f"Valid   - loss: {val_loss:.4f}, psnr: {val_psnr:.2f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val": best_val,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        save_checkpoint(state, Path(cfg.out_dir), is_best=is_best, epoch=epoch)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()


