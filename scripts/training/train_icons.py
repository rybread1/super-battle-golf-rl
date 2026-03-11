"""Training script for the icon/object detection CNN.

Usage:
    uv run python scripts/training/train_icons.py --data screenshots/cnn_training/annotations.json --name v1
    uv run python scripts/training/train_icons.py --data annotations.json --name v2 --epochs 50
    uv run python scripts/training/train_icons.py --data annotations.json --resume checkpoints/icon_net/v1.pt

TensorBoard:
    uv run tensorboard --logdir runs/icon_net
"""

import argparse
import pathlib
import time

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from sbg.models.icon_net import IconNet, TARGETS
from sbg.models.dataset import IconDataset
from sbg.models.loss import icon_loss


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = IconDataset(args.data, augment=True)
    print(f"Loaded {len(dataset)} annotated frames")

    # Train/val split (80/20)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # Disable augmentation for validation
    val_set.dataset = IconDataset(args.data, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    model = IconNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    save_dir = pathlib.Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from {args.resume} (epoch {checkpoint['epoch']}, "
              f"best_val={best_val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e})")

    # TensorBoard
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    run_name = args.name or f"icon_net_{time.strftime('%Y%m%d_%H%M%S')}"
    tb_dir = project_root / "runs" / "icon_net" / run_name
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: {tb_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = {key: batch[key].to(device) for key in TARGETS}

            pred = model(images)
            loss = icon_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        coord_err_sum = {key: 0.0 for key in TARGETS}
        coord_err_n = {key: 0 for key in TARGETS}
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = {key: batch[key].to(device) for key in TARGETS}

                pred = model(images)
                loss = icon_loss(pred, targets)
                val_loss += loss.item() * images.size(0)

                for key in TARGETS:
                    t = targets[key]
                    p = pred[key]
                    mask = t[:, 0].bool()
                    if mask.any():
                        px = p[mask, 1] * 1280
                        py = p[mask, 2] * 720
                        tx = t[mask, 1] * 1280
                        ty = t[mask, 2] * 720
                        dist = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2)
                        coord_err_sum[key] += dist.sum().item()
                        coord_err_n[key] += mask.sum().item()

        val_loss /= n_val
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        coord_err = {}
        for key in TARGETS:
            if coord_err_n[key] > 0:
                coord_err[key] = coord_err_sum[key] / coord_err_n[key]
            else:
                coord_err[key] = None

        # --- Log to TensorBoard ---
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("lr", lr, epoch)
        for key in TARGETS:
            if coord_err[key] is not None:
                writer.add_scalar(f"coord_error_px/{key}", coord_err[key], epoch)

        # --- Console ---
        err_parts = []
        for key in TARGETS:
            if coord_err[key] is not None:
                err_parts.append(f"{key}={coord_err[key]:.0f}px")
            else:
                err_parts.append(f"{key}=n/a")
        err_str = "  ".join(err_parts)
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"{err_str}  lr={lr:.2e}")

        writer.flush()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                             save_dir / f"{run_name}_best.pt")
            print(f"  -> saved best (val_loss={val_loss:.4f})")

    # Save final
    _save_checkpoint(model, optimizer, scheduler, args.epochs, best_val_loss,
                     save_dir / f"{run_name}_final.pt")
    writer.close()
    print(f"\nTraining complete. Checkpoints saved to {save_dir}/")


def _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train icon/object detection CNN")
    parser.add_argument("--data", required=True,
                        help="Path to annotations JSON file")
    parser.add_argument("--name", type=str, default=None,
                        help="Checkpoint name (used for filenames and TensorBoard run)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5,
                        help="LR scheduler patience (epochs without val improvement)")
    parser.add_argument("--output", default="checkpoints/icon_net",
                        help="Directory to save checkpoints")
    args = parser.parse_args()
    train(args)
