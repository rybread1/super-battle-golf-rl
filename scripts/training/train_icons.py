"""Training script for the icon/object detection CNN.

Usage:
    uv run python scripts/training/train_icons.py --data screenshots/cnn_train/annotations.json
    uv run python scripts/training/train_icons.py --data annotations.json --epochs 50
"""

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader, random_split

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float("inf")
    save_dir = pathlib.Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
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
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = {key: 0 for key in TARGETS}
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = {key: batch[key].to(device) for key in TARGETS}

                pred = model(images)
                loss = icon_loss(pred, targets)
                val_loss += loss.item() * images.size(0)

                for key in TARGETS:
                    pred_present = (torch.sigmoid(pred[key][:, 0]) > 0.5).float()
                    actual = targets[key][:, 0]
                    correct[key] += (pred_present == actual).sum().item()
                total += images.size(0)

        val_loss /= n_val
        acc_str = "  ".join(
            f"{key}={correct[key] / total * 100:.0f}%" for key in TARGETS
        )

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"{acc_str}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "icon_net_best.pt")
            print(f"  -> saved best model (val_loss={val_loss:.4f})")

    # Save final
    torch.save(model.state_dict(), save_dir / "icon_net_final.pt")
    print(f"\nTraining complete. Models saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train icon/object detection CNN")
    parser.add_argument("--data", required=True,
                        help="Path to annotations JSON file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="checkpoints/icon_net",
                        help="Directory to save model weights")
    args = parser.parse_args()
    train(args)
