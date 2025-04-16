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
import shutil
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
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def get_sampler_targets(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def clean_codecarbon_lock():
    """Διαγράφει το αρχείο κλειδώματος του codecarbon αν υπάρχει."""
    if os.path.exists(CODECARBON_LOCK_FILE):
        try:
            os.remove(CODECARBON_LOCK_FILE)
            print(f"✅ Removed stale codecarbon lock file: {CODECARBON_LOCK_FILE}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to remove codecarbon lock file: {str(e)}")
            return False
    return True

def log_gpu_power(log_file, stop_event):
    """Συνεχώς καταγράφει την κατανάλωση ενέργειας της GPU σε ένα αρχείο."""
    while not stop_event.is_set():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader'], capture_output=True, text=True, check=True)
            power_usage = result.stdout.strip()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(log_file, 'a') as f:
                f.write(f"{timestamp},{power_usage}\n")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error running nvidia-smi: {e}")
            break
        except Exception as e:
            print(f"⚠️ An error occurred during GPU power logging: {e}")
            break
        time.sleep(5)  

def train_model(config):
    
    results_path = config['results_path']
    if not os.path.exists(results_path):
        print(f"Creating results directory: {results_path}")
        os.makedirs(results_path, exist_ok=True)
    else:
        print(f"Results directory exists: {results_path}")

    
    try:
        test_file_path = os.path.join(results_path, "test_write_permissions.txt")
        with open(test_file_path, 'w') as f:
            f.write("Test write permissions")
        os.remove(test_file_path)
        print(f"✅ Write permissions confirmed for: {results_path}")
    except Exception as e:
        print(f"❌ No write permissions for {results_path}: {str(e)}")
        print("Please check directory permissions or specify a different path.")
        sys.exit(1)

    
    lock_removed = clean_codecarbon_lock()

    tracker = None
    try:
        
        print("Initializing EmissionsTracker...")
        emissions_file = os.path.join(results_path, "emissions.csv")
        print(f"Emissions will be saved to: {emissions_file}")

        if lock_removed:
            tracker = EmissionsTracker(
                output_dir=results_path,
                output_file="emissions.csv",
                measure_power_secs=15,
                log_level="warning",  
                save_to_file=True,  
                allow_multiple_runs=True  
            )
            print("Starting EmissionsTracker...")
            tracker.start()
            print("✅ EmissionsTracker started successfully")
        else:
            print("⚠️ Unable to remove codecarbon lock file. Energy tracking will be disabled.")
    except Exception as e:
        print(f"\n⚠️ CodeCarbon Tracker error: {str(e)}")
        user_input = input("❓ Do you want to continue training without energy tracking? [y/n]: ")
        if user_input.lower() != 'y':
            print("⛔ Exiting training.")
            sys.exit(1)
        tracker = None
        print("Continuing without energy tracking")

    
    with open(os.path.join(results_path, "training_start.txt"), "w") as f:
        f.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        f.write(f"EmissionsTracker active: {tracker is not None}\n")

    
    gpu_log_file = os.path.join(results_path, "gpu_power.log")
    stop_gpu_log_event = threading.Event()
    gpu_log_thread = threading.Thread(target=log_gpu_power, args=(gpu_log_file, stop_gpu_log_event))
    gpu_log_thread.start()
    print(f"📊 GPU power usage will be logged to: {gpu_log_file}")

    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    image_size = config["image_size"]
    learning_rate = config["learning_rate"]
    num_workers = config["num_workers"]
    save_path = config["save_path"]
    loss_function = config["loss_function"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    print(f"Loading datasets from {data_dir}...")
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images")

    sampler = get_sampler_targets(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    
    if config["pretrained"]:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    if loss_function == "focal":
        criterion = FocalLoss()
        print("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Cross Entropy Loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    accs = []

    try:
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total
            train_losses.append(train_loss)
            accs.append(train_acc)

            model.eval()
            val_loss = 0.0
            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_losses.append(val_loss / val_total)
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_losses[-1]:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                temp_save_path = os.path.join(results_path, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), temp_save_path)
                print(f"Saved intermediate model to {temp_save_path}")

        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved final model to {save_path}")

        y_true = []
        y_pred = []
        model.eval()
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        class_report = classification_report(y_true, y_pred, target_names=val_dataset.classes, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)

        print("Saving evaluation metrics and plots...")
        with open(os.path.join(results_path, "metrics.yaml"), "w") as f:
            yaml.dump(class_report, f)

        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title("Loss Curves")
        plt.savefig(os.path.join(results_path, "loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(accs, label='Train Accuracy')
        plt.legend()
        plt.title("Training Accuracy")
        plt.savefig(os.path.join(results_path, "accuracy_curve.png"))
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(val_dataset.classes))
        plt.xticks(tick_marks, labels=val_dataset.classes, rotation=90)
        plt.yticks(tick_marks, labels=val_dataset.classes)

        # Προσθέστε τις ετικέτες με τους αριθμούς μέσα στα κελιά
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, f'{conf_matrix[i, j]}',
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(results_path, "confusion_matrix.png"))
        plt.close()

        print("✅ Training complete. Metrics and model saved.")

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        
        error_save_path = os.path.join(results_path, "model_error_checkpoint.pth")
        torch.save(model.state_dict(), error_save_path)
        print(f"Saved error checkpoint to {error_save_path}")
        raise e
    finally:
        
        stop_gpu_log_event.set()
        gpu_log_thread.join()
        print(f"🛑 GPU power logging stopped.")

        
        if tracker:
            print("Stopping EmissionsTracker and saving energy data...")
            try:
                emissions_data = tracker.stop()
                print(f"Energy tracking complete. Emissions data: {emissions_data}")
                emissions_file = os.path.join(results_path, "emissions.csv")
                if os.path.exists(emissions_file):
                    print(f"✅ Emissions data saved to {emissions_file}")
                else:
                    print(f"❌ Warning: Emissions file not found at {emissions_file}")
                    if emissions_data:
                        with open(emissions_file, "w") as f:
                            f.write(f"timestamp,emissions_data\n{time.strftime('%Y-%m-%d %H:%M:%S')},{emissions_data}\n")
                        print(f"✅ Manually created emissions file at {emissions_file}")
            except Exception as e:
                print(f"❌ Error stopping tracker: {str(e)}")

            clean_codecarbon_lock()

        
        try:
            with open(os.path.join(results_path, "training_end.txt"), "w") as f:
                f.write(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                if train_losses:
                    f.write(f"Final train loss: {train_losses[-1]}\n")
                if val_losses:
                    f.write(f"Final validation loss: {val_losses[-1]}\n")
                if accs:
                    f.write(f"Final train accuracy: {accs[-1]}\n")
        except Exception as e:
            print(f"❌ Error saving training end stats: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
        train_model(config)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
