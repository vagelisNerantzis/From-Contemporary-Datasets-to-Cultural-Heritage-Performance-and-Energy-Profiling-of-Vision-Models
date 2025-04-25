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

# Define the path to the codecarbon lock file.
# This file is used by codecarbon to prevent multiple instances from running simultaneously.
CODECARBON_LOCK_FILE = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", ".codecarbon.lock")

class FocalLoss(nn.Module):
    """
    Implements the Focal Loss function for classification tasks.
    Focal Loss is designed to address class imbalance by down-weighting
    the loss for well-classified examples and focusing on hard, misclassified ones.

    Args:
        gamma (float): Focusing parameter. Controls the rate at which easy examples are down-weighted.
        alpha (float): Weighting factor for each class.
    """
    def __init__(self, gamma=2, alpha=1):
        # Call the constructor of the parent class (nn.Module).
        super(FocalLoss, self).__init__()
        # Store the gamma parameter.
        self.gamma = gamma
        # Store the alpha parameter.
        self.alpha = alpha
        # Initialize the standard Cross-Entropy Loss with no reduction.
        # 'reduction='none'' computes the loss per element in the batch, needed for weighting.
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        """
        Computes the Focal Loss.

        Args:
            inputs (torch.Tensor): The output logits from the model (before softmax).
                                   Shape: (batch_size, num_classes)
            targets (torch.Tensor): The ground truth class labels.
                                    Shape: (batch_size,)

        Returns:
            torch.Tensor: The computed mean Focal Loss over the batch.
        """
        # Compute the standard Cross-Entropy Loss for each sample.
        ce_loss = self.ce(inputs, targets)
        # Calculate the probability of the predicted class (pt).
        pt = torch.exp(-ce_loss)
        # Compute the Focal Loss: alpha * (1 - pt)^gamma * CE_loss.
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # Return the mean of the focal loss over the batch.
        return focal_loss.mean()

def get_sampler_targets(dataset):
    """
    Creates a WeightedRandomSampler to handle class imbalance in the dataset.
    Assigns higher weights to samples from minority classes.

    Args:
        dataset (torch.utils.data.Dataset): The dataset. Assumes dataset.samples is (data, label).

    Returns:
        torch.utils.data.WeightedRandomSampler: The sampler object.
    """
    # Extract the target labels for each sample.
    targets = [s[1] for s in dataset.samples]
    # Count the occurrences of each class.
    class_counts = Counter(targets)
    # Calculate inverse weights for each sample based on class frequency.
    weights = [1.0 / class_counts[t] for t in targets]
    # Create and return the weighted sampler.
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def clean_codecarbon_lock():
    """
    Deletes the codecarbon lock file if it exists.
    Handles cases where a previous run might have crashed and left the lock file.

    Returns:
        bool: True if the lock file was removed or didn't exist, False otherwise.
    """
    # Check if the lock file exists.
    if os.path.exists(CODECARBON_LOCK_FILE):
        try:
            # Attempt to remove the file.
            os.remove(CODECARBON_LOCK_FILE)
            print("✅ Removed stale codecarbon lock file.")
            return True
        except Exception as e:
            # Print an error if removal fails.
            print(f"⚠️ Failed to remove lock file: {e}")
            return False
    # Return True if the file didn't exist.
    return True

def log_gpu_power(log_file, stop_event):
    """
    Continuously logs the GPU power consumption using nvidia-smi in a separate thread.

    Args:
        log_file (str): Path to the file for logging.
        stop_event (threading.Event): Event to signal the thread to stop.
    """
    # Loop until the stop event is set.
    while not stop_event.is_set():
        try:
            # Run nvidia-smi command to get power draw.
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader'],
                                    capture_output=True, text=True, check=True)
            # Extract power usage.
            power_usage = result.stdout.strip()
            # Get current timestamp.
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            # Append timestamp and power usage to the log file.
            with open(log_file, 'a') as f:
                f.write(f"{timestamp},{power_usage}\n")
        except Exception as e:
            # Print error if logging fails and break loop.
            print(f"⚠️ GPU logging error: {e}")
            break
        # Wait for 5 seconds before the next query.
        time.sleep(5)

def train_model(config):
    """
    Main function to train the Swin Transformer Tiny model.
    Handles setup, data loading, training, evaluation, and saving results.

    Args:
        config (dict): Configuration dictionary loaded from a YAML file.
                       Contains parameters like data_dir, batch_size, num_epochs, etc.
    """
    # Get the results directory path from config.
    results_path = config["results_path"]
    # Create the results directory if it doesn't exist.
    os.makedirs(results_path, exist_ok=True)

    # --- Check Write Permissions ---
    # Verify write access to the results directory.
    try:
        # Attempt to create and delete a test file.
        with open(os.path.join(results_path, "test.txt"), "w") as f:
            f.write("test")
        os.remove(os.path.join(results_path, "test.txt"))
    except Exception as e:
        # Print error and exit if write access is denied.
        print(f"❌ No write access to {results_path}: {e}")
        sys.exit(1)

    # --- CodeCarbon Tracker Initialization ---
    # Clean up the codecarbon lock file before initializing the tracker.
    lock_removed = clean_codecarbon_lock()
    tracker = None # Initialize tracker to None.
    try:
        # Initialize and start the tracker if the lock file was removed or didn't exist.
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
            # Skip tracking if the lock file is present.
            print("⚠️ CodeCarbon lock file present. Skipping tracking.")
    except Exception as e:
        # Handle tracker initialization errors.
        print(f"⚠️ CodeCarbon error: {e}")
        # Ask user if they want to continue without tracking.
        if input("Continue without energy tracking? [y/n]: ").lower() != "y":
            # Exit if user says no.
            sys.exit(1)

    # --- Start GPU Power Logging Thread ---
    # Define GPU log file path.
    gpu_log_file = os.path.join(results_path, "gpu_power.log")
    # Create stop event for the thread.
    stop_event = threading.Event()
    # Create and start the GPU logging thread.
    thread = threading.Thread(target=log_gpu_power, args=(gpu_log_file, stop_event))
    thread.start()

    # --- Device Configuration ---
    # Determine the device to use (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Transformations ---
    # Define image transformations (resize and convert to tensor).
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor()
    ])

    # --- Dataset Loading ---
    # Load training and validation datasets using ImageFolder.
    train_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "val"), transform=transform)

    # --- Sampler for Class Imbalance ---
    # Get the weighted sampler for the training dataset.
    sampler = get_sampler_targets(train_dataset)
    # --- Data Loaders ---
    # Create data loaders for efficient batching.
    # train_loader uses the weighted sampler.
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=config["num_workers"])
    # val_loader does not shuffle.
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    # --- Model Initialization ---
    # Initialize the Swin Transformer Tiny model.
    if config["pretrained"]:
        # Load with pre-trained ImageNet weights.
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    else:
        # Initialize with random weights.
        model = models.swin_t(weights=None)

    # --- Modify the Classifier Head ---
    # Replace the final classification layer to match the number of classes.
    model.head = nn.Linear(model.head.in_features, len(train_dataset.classes))
    # Move the model to the selected device.
    model = model.to(device)

    # --- Loss Function Selection ---
    # Select the loss function based on config ('focal' or 'crossentropy').
    criterion = FocalLoss() if config["loss_function"] == "focal" else nn.CrossEntropyLoss()
    # --- Optimizer Configuration ---
    # Initialize the Adam optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # --- Lists to Store Metrics ---
    # Initialize lists to store training/validation losses and training accuracies per epoch.
    train_losses, val_losses, accs = [], [], []

    # --- Training Loop ---
    # Iterate over the specified number of epochs.
    for epoch in range(config["num_epochs"]):
        # Set model to training mode.
        model.train()
        # Initialize metrics for the epoch.
        running_loss, correct, total = 0.0, 0, 0
        # Iterate through the training data loader.
        for inputs, labels in train_loader:
            # Move data to device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero gradients.
            optimizer.zero_grad()
            # Forward pass.
            outputs = model(inputs)
            # Compute loss.
            loss = criterion(outputs, labels)
            # Backward pass (compute gradients).
            loss.backward()
            # Update model weights.
            optimizer.step()
            # Accumulate loss.
            running_loss += loss.item() * inputs.size(0)
            # Count correct predictions.
            correct += (outputs.argmax(1) == labels).sum().item()
            # Update total samples.
            total += labels.size(0)
        # Calculate and store average training loss and accuracy for the epoch.
        train_losses.append(running_loss / total)
        accs.append(correct / total)

        # --- Validation Phase ---
        # Set model to evaluation mode.
        model.eval()
        # Initialize validation metrics.
        val_loss, val_correct, val_total = 0.0, 0, 0
        # Disable gradient calculation.
        with torch.no_grad():
            # Iterate through validation data loader.
            for inputs, labels in val_loader:
                # Move data to device.
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass.
                outputs = model(inputs)
                # Compute loss.
                loss = criterion(outputs, labels)
                # Accumulate loss.
                val_loss += loss.item() * inputs.size(0)
                # Count correct predictions.
                val_correct += (outputs.argmax(1) == labels).sum().item()
                # Update total samples.
                val_total += labels.size(0)
        # Calculate and store average validation loss for the epoch.
        val_losses.append(val_loss / val_total)
        # Print epoch progress and metrics.
        print(f"[{epoch+1}/{config['num_epochs']}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Train Acc: {accs[-1]:.4f} | Val Acc: {val_correct/val_total:.4f}")

    # --- Save Final Model ---
    # Create directory for save path if needed.
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    # Save the model's state dictionary.
    torch.save(model.state_dict(), config["save_path"])
    print(f"✅ Model saved to {config['save_path']}")

    # --- Evaluation on Validation Set ---
    # Initialize lists for true and predicted labels.
    y_true, y_pred = [], []
    # Set model to evaluation mode.
    model.eval()
    # Iterate through validation data without gradients.
    for inputs, labels in val_loader:
        # Move data to device.
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass.
        outputs = model(inputs)
        # Extend true labels list.
        y_true.extend(labels.cpu().numpy())
        # Extend predicted labels list (get class index with highest score).
        y_pred.extend(outputs.argmax(1).cpu().numpy())

    # --- Save Evaluation Metrics ---
    # Save classification report to a YAML file.
    with open(os.path.join(results_path, "metrics.yaml"), "w") as f:
        # Generate classification report.
        report = classification_report(
            y_true,
            y_pred,
            target_names=val_dataset.classes,
            output_dict=True,
            zero_division=0 # Handle cases with no samples in a class.
        )
        # Dump report to YAML.
        yaml.dump(report, f)

    # --- Plot and Save Loss Curves ---
    plt.figure() # Create a new figure.
    plt.plot(train_losses, label='Train Loss') # Plot training loss.
    plt.plot(val_losses, label='Val Loss')     # Plot validation loss.
    plt.title("Loss Curves") # Set title.
    plt.legend() # Show legend.
    plt.savefig(os.path.join(results_path, "loss_curve.png")) # Save plot.
    plt.close() # Close figure.

    # --- Plot and Save Accuracy Curve ---
    plt.figure() # Create a new figure.
    plt.plot(accs, label='Train Accuracy') # Plot training accuracy.
    plt.title("Accuracy Curve") # Set title.
    plt.legend() # Show legend.
    plt.savefig(os.path.join(results_path, "accuracy_curve.png")) # Save plot.
    plt.close() # Close figure.

    # --- Plot and Save Confusion Matrix ---
    conf_matrix = confusion_matrix(y_true, y_pred) # Compute confusion matrix.
    plt.figure(figsize=(10, 8)) # Create a new figure with specified size.
    plt.imshow(conf_matrix, cmap='Blues') # Display matrix as image.
    plt.title("Confusion Matrix") # Set title.
    plt.colorbar() # Add color bar.
    # Set ticks and labels for axes using class names.
    ticks = np.arange(len(val_dataset.classes))
    plt.xticks(ticks, val_dataset.classes, rotation=90)
    plt.yticks(ticks, val_dataset.classes)
    # Add text labels inside cells.
    thresh = conf_matrix.max() / 2. # Threshold for text color.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            # Get cell value, handling potential tensor format.
            value = conf_matrix[i, j].item() if hasattr(conf_matrix[i, j], 'item') else conf_matrix[i, j]
            # Add text with value and appropriate color.
            plt.text(j, i, f'{value}',
                     horizontalalignment="center",
                     color="white" if value > thresh else "black")
    plt.tight_layout() # Adjust layout.
    plt.savefig(os.path.join(results_path, "confusion_matrix.png")) # Save plot.
    plt.close() # Close figure.

    # --- Cleanup ---
    # Signal GPU logging thread to stop and wait for it to finish.
    stop_event.set()
    thread.join()
    # Stop the CodeCarbon tracker if it was started.
    if tracker:
        tracker.stop()

# --- Main Execution Block ---
# This block runs when the script is executed directly.
if __name__ == "__main__":
    # --- Argument Parsing ---
    # Create an argument parser.
    parser = argparse.ArgumentParser()
    # Add argument for the config file path.
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    # Parse arguments.
    args = parser.parse_args()
    # --- Load Configuration and Start Training ---
    # Open and load the YAML config file.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # Call the main training function.
    train_model(config)
