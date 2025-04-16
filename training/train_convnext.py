import os
import yaml
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse
import sys
import time
import subprocess
import threading

CODECARBON_LOCK_FILE = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", ".codecarbon.lock")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

def get_sampler_targets(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def clean_codecarbon_lock():
    if os.path.exists(CODECARBON_LOCK_FILE):
        try:
            os.remove(CODECARBON_LOCK_FILE)
            print("✅ Removed stale codecarbon lock file.")
            return True
        except Exception as e:
            print(f"⚠️ Failed to remove lock file: {e}")
            return False
    return True

def log_gpu_power(log_file, stop_event):
    while not stop_event.is_set():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader'],
                                    capture_output=True, text=True, check=True)
            power_usage = result.stdout.strip()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(log_file, 'a') as f:
                f.write(f"{timestamp},{power_usage}\n")
        except Exception as e:
            print(f"⚠️ GPU logging error: {e}")
            break
        time.sleep(5)

def train_model(config):
    results_path = config["results_path"]
    os.makedirs(results_path, exist_ok=True)

    try:
        with open(os.path.join(results_path, "test.txt"), "w") as f:
            f.write("test")
        os.remove(os.path.join(results_path, "test.txt"))
    except Exception as e:
        print(f"❌ No write access to {results_path}: {e}")
        sys.exit(1)

    lock_removed = clean_codecarbon_lock()
    tracker = None
    try:
        if lock_removed:
            tracker = EmissionsTracker(
                output_dir=results_path,
                output_file="emissions.csv",
                measure_power_secs=15,
                log_level="warning",
                save_to_file=True,
                allow_multiple_runs=True
            )
            tracker.start()
            print("✅ CodeCarbon tracker started.")
        else:
            print("⚠️ CodeCarbon lock file present. Skipping tracking.")
    except Exception as e:
        print(f"⚠️ CodeCarbon error: {e}")
        if input("Continue without energy tracking? [y/n]: ").lower() != "y":
            sys.exit(1)

    gpu_log_file = os.path.join(results_path, "gpu_power.log")
    stop_event = threading.Event()
    thread = threading.Thread(target=log_gpu_power, args=(gpu_log_file, stop_event))
    thread.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "val"), transform=transform)

    sampler = get_sampler_targets(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    if config["pretrained"]:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    else:
        model = models.convnext_tiny(weights=None)

    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = FocalLoss() if config["loss_function"] == "focal" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_losses, val_losses, accs = [], [], []

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_losses.append(running_loss / total)
        accs.append(correct / total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_losses.append(val_loss / val_total)
        print(f"[{epoch+1}/{config['num_epochs']}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Train Acc: {accs[-1]:.4f} | Val Acc: {val_correct/val_total:.4f}")

    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["save_path"])
    print(f"✅ Model saved to {config['save_path']}")

    y_true, y_pred = [], []
    model.eval()
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.argmax(1).cpu().numpy())

    with open(os.path.join(results_path, "metrics.yaml"), "w") as f:
        yaml.dump(
            classification_report(
                y_true,
                y_pred,
                target_names=val_dataset.classes,
                output_dict=True,
                zero_division=0
            ),
            f
        )

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(results_path, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(accs, label='Train Accuracy')
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(results_path, "accuracy_curve.png"))
    plt.close()

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(val_dataset.classes))
    plt.xticks(ticks, val_dataset.classes, rotation=90)
    plt.yticks(ticks, val_dataset.classes)
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = conf_matrix[i, j].item() if hasattr(conf_matrix[i, j], 'item') else conf_matrix[i, j]
            plt.text(j, i, f'{value}',
                     horizontalalignment="center",
                     color="white" if value > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "confusion_matrix.png"))
    plt.close()

    stop_event.set()
    thread.join()
    if tracker:
        tracker.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    train_model(config)
