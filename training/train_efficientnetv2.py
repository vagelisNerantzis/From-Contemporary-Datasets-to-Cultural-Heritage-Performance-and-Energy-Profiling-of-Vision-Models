# training/train_efficientnetv2.py

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
    """
    Implements the Focal Loss function.
    This loss function is useful for classification tasks with severe class imbalance.
    It down-weights easy examples and focuses training on hard examples.

    Args:
        gamma (float): Focusing parameter. Controls the rate at which easy examples are down-weighted.
        alpha (float): Weighting factor for each class. Can be a scalar or a list/tensor
                       of weights per class (though implemented as a scalar here).
    """
    def __init__(self, gamma=2, alpha=1):
        # Call the constructor of the parent class (nn.Module).
        super(FocalLoss, self).__init__()
        # Store the gamma parameter.
        self.gamma = gamma
        # Store the alpha parameter.
        self.alpha = alpha
        # Initialize the standard Cross-Entropy Loss with no reduction.
        # 'reduction='none'' means the loss is computed for each element in the batch separately.
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        """
        Computes the Focal Loss for a batch of inputs and targets.

        Args:
            inputs (torch.Tensor): The raw, unnormalized scores (logits) from the model.
                                   Shape: (batch_size, num_classes)
            targets (torch.Tensor): The ground truth class labels.
                                    Shape: (batch_size,)

        Returns:
            torch.Tensor: The mean Focal Loss over the batch.
        """
        # Compute the standard Cross-Entropy Loss for each sample.
        ce_loss = self.ce(inputs, targets)
        # Calculate the probability of the target class using the negative exponent of the CE loss.
        # pt = exp(-CE_loss) - This is equivalent to the predicted probability if CE_loss was -log(p_t).
        # A higher pt means the model is more confident and correct.
        pt = torch.exp(-ce_loss)
        # Compute the Focal Loss: alpha * (1 - pt)^gamma * CE_loss.
        # The term (1 - pt)^gamma is the modulating factor. It becomes smaller for
        # well-classified examples (high pt), reducing their contribution to the loss,
        # and larger for misclassified or hard examples (low pt), increasing their contribution.
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # Return the mean of the focal loss over the batch.
        return focal_loss.mean()

def get_sampler_targets(dataset):
    """
    Creates a WeightedRandomSampler to address class imbalance in a dataset.
    It calculates weights inversely proportional to class frequencies and uses
    these weights to sample data points during training, ensuring that
    batches are more balanced in terms of class representation.

    Args:
        dataset (torch.utils.data.Dataset): The dataset object. Assumes it has a
                                           `.samples` attribute, where each element
                                           is a tuple (data, label).

    Returns:
        torch.utils.data.WeightedRandomSampler: The sampler object.
    """
    # Extract the target labels from the dataset samples.
    targets = [s[1] for s in dataset.samples]
    # Count the occurrences of each class label.
    class_counts = Counter(targets)
    # Calculate weights for each sample. Weight = 1.0 / count_of_sample's_class.
    # Samples from less frequent classes get higher weights.
    weights = [1.0 / class_counts[t] for t in targets]
    # Create a WeightedRandomSampler.
    # weights: The list of calculated weights.
    # num_samples: The total number of samples to draw per epoch (set to the dataset size).
    # replacement=True: Samples are drawn with replacement. This is necessary for
    #                   oversampling minority classes using weights.
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def clean_codecarbon_lock():
    """
    Deletes the codecarbon lock file if it exists.
    This is a helper function to clean up potential leftover lock files
    from previous unsuccessful runs that might prevent a new tracker from starting.

    Returns:
        bool: True if the lock file was successfully removed or did not exist,
              False otherwise.
    """
    # Check if the codecarbon lock file exists at the defined path.
    if os.path.exists(CODECARBON_LOCK_FILE):
        try:
            # Attempt to remove the file.
            os.remove(CODECARBON_LOCK_FILE)
            # Print a success message.
            print("✅ Removed stale codecarbon lock file.")
            return True
        except Exception as e:
            # Print an error message if removal fails.
            print(f"⚠️ Failed to remove lock file: {e}")
            return False
    # Return True if the file didn't exist.
    return True

def log_gpu_power(log_file, stop_event):
    """
    Continuously logs the GPU power consumption using nvidia-smi.
    This function is intended to run in a separate thread during training
    to monitor energy usage in real-time.

    Args:
        log_file (str): The path to the file where power data will be appended.
        stop_event (threading.Event): An event flag to signal the thread to stop.
    """
    # Loop until the stop_event is set.
    while not stop_event.is_set():
        try:
            # Execute the nvidia-smi command to get GPU power draw.
            # '--query-gpu=power.draw': Specify the metric to query.
            # '--format=csv,noheader': Output in CSV format without a header.
            # capture_output=True: Capture the command's output.
            # text=True: Decode output as text.
            # check=True: Raise an exception if the command fails.
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader'],
                                    capture_output=True, text=True, check=True)
            # Extract and clean up the power usage reading.
            power_usage = result.stdout.strip()
            # Get the current timestamp.
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            # Open the log file in append mode ('a') and write the timestamp and power usage.
            with open(log_file, 'a') as f:
                f.write(f"{timestamp},{power_usage}\n")
        except Exception as e:
            # Print an error message if GPU logging fails and break the loop.
            print(f"⚠️ GPU logging error: {e}")
            break
        # Pause for 5 seconds before the next query.
        time.sleep(5)

def train_model(config):
    """
    Manages the entire training process for the EfficientNetV2 model.
    This includes setting up directories, initializing the tracker, loading data,
    building and configuring the model, running the training and validation loops,
    saving the model and metrics, and stopping energy tracking.

    Args:
        config (dict): A dictionary containing all necessary training parameters
                       loaded from a YAML configuration file.
    """
    # Get the path to the directory where training results will be saved.
    results_path = config["results_path"]
    # Create the results directory if it doesn't already exist.
    os.makedirs(results_path, exist_ok=True) # exist_ok=True prevents an error if the directory already exists.

    # --- Check Write Permissions ---
    # Verify that the script has write permissions in the results directory.
    try:
        # Attempt to create and immediately delete a test file in the results directory.
        with open(os.path.join(results_path, "test.txt"), "w") as f:
            f.write("test")
        os.remove(os.path.join(results_path, "test.txt"))
    except Exception as e:
        # If an error occurs, print a message and exit the script.
        print(f"❌ No write access to {results_path}: {e}")
        sys.exit(1)

    # --- CodeCarbon Tracker Initialization ---
    # Attempt to clean the codecarbon lock file.
    lock_removed = clean_codecarbon_lock()
    # Initialize the tracker variable to None.
    tracker = None
    try:
        # If the lock file was successfully removed (or didn't exist), initialize the tracker.
        if lock_removed:
            tracker = EmissionsTracker(
                output_dir=results_path,         # Directory to save emissions data.
                output_file="emissions.csv",     # Name of the output file.
                measure_power_secs=15,           # Frequency of power measurement.
                log_level="warning",             # Set log level to warning.
                save_to_file=True,               # Enable saving to a file.
                allow_multiple_runs=True         # Allow appending to the same file across multiple runs.
            )
            # Start the emissions tracker.
            tracker.start()
            # Print a success message.
            print("✅ CodeCarbon tracker started.")
        else:
            # If the lock file was present and couldn't be removed, skip tracking.
            print("⚠️ CodeCarbon lock file present. Skipping tracking.")
    except Exception as e:
        # If any error occurs during tracker initialization or start, print an error.
        print(f"⚠️ CodeCarbon error: {e}")
        # Ask the user if they want to continue training without energy tracking.
        if input("Continue without energy tracking? [y/n]: ").lower() != "y":
            # If the user says no, exit the script.
            sys.exit(1)

    # --- Start GPU Power Logging Thread ---
    # Define the path for the GPU power log file.
    gpu_log_file = os.path.join(results_path, "gpu_power.log")
    # Create an event to signal the GPU logging thread to stop.
    stop_event = threading.Event()
    # Create a new thread that will run the log_gpu_power function.
    thread = threading.Thread(target=log_gpu_power, args=(gpu_log_file, stop_event))
    # Start the GPU logging thread.
    thread.start()

    # --- Device Configuration ---
    # Determine the device to use for training (CUDA if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Transformations ---
    # Define the image transformations to apply to the dataset.
    transform = transforms.Compose([
        # Resize images to the specified image_size from the config.
        transforms.Resize((config["image_size"], config["image_size"])),
        # Convert images to PyTorch tensors (scales pixel values to [0, 1]).
        transforms.ToTensor()
    ])

    # --- Dataset Loading ---
    # Load the training and validation datasets using ImageFolder.
    # ImageFolder expects the data_dir to contain subdirectories for each class.
    train_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "val"), transform=transform)

    # --- Sampler for Class Imbalance ---
    # Get the weighted random sampler for the training dataset to handle class imbalance.
    sampler = get_sampler_targets(train_dataset)
    # --- Data Loaders ---
    # Create DataLoaders for efficient batching and loading of the datasets.
    # train_loader uses the weighted sampler for balanced batching.
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=config["num_workers"])
    # val_loader uses a fixed batch size and does not shuffle the data.
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    # --- Model Initialization ---
    # Initialize the EfficientNetV2-S model.
    if config["pretrained"]:
        # If 'pretrained' is true in the config, load pre-trained weights from ImageNet.
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    else:
        # Otherwise, initialize the model with random weights.
        model = models.efficientnet_v2_s(weights=None)

    # --- Modify the Classifier Head ---
    # Replace the final classification layer to match the number of classes in the dataset.
    # EfficientNetV2's classifier is typically a sequential block, and the linear layer is the second element (index 1).
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes))
    # Move the model to the selected device (GPU or CPU).
    model = model.to(device)

    # --- Loss Function Selection ---
    # Choose the loss function based on the 'loss_function' setting in the config.
    # Use FocalLoss if specified, otherwise use standard Cross-Entropy Loss.
    criterion = FocalLoss() if config["loss_function"] == "focal" else nn.CrossEntropyLoss()
    # --- Optimizer Configuration ---
    # Initialize the Adam optimizer with the model's parameters and the specified learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # --- Lists to Store Metrics ---
    # Initialize lists to store loss and accuracy values for plotting and tracking.
    train_losses, val_losses, accs = [], [], []

    # --- Training Loop ---
    # Iterate over the specified number of training epochs.
    for epoch in range(config["num_epochs"]):
        # Set the model to training mode. Enables gradient calculation, dropout, batch norm updates.
        model.train()
        # Initialize running loss, correct predictions, and total samples for the current epoch.
        running_loss, correct, total = 0.0, 0, 0
        # Iterate through the training data loader to get batches.
        for inputs, labels in train_loader:
            # Move data to the selected device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero out the gradients accumulated from the previous iteration.
            optimizer.zero_grad()
            # Perform a forward pass to get model outputs (logits).
            outputs = model(inputs)
            # Compute the loss using the chosen criterion.
            loss = criterion(outputs, labels)
            # Perform a backward pass to compute gradients.
            loss.backward()
            # Update model parameters based on gradients.
            optimizer.step()
            # Accumulate the loss for the current batch.
            running_loss += loss.item() * inputs.size(0)
            # Count correct predictions in the current batch.
            correct += (outputs.argmax(1) == labels).sum().item()
            # Add the number of samples in the current batch to the total.
            total += labels.size(0)
        # Calculate and store the average training loss for the epoch.
        train_losses.append(running_loss / total)
        # Calculate and store the training accuracy for the epoch.
        accs.append(correct / total)

        # --- Validation Phase ---
        # Set the model to evaluation mode. Disables dropout, uses batch norm statistics.
        model.eval()
        # Initialize validation loss, correct predictions, and total samples for validation.
        val_loss, val_correct, val_total = 0.0, 0, 0
        # Disable gradient calculations during validation (inference only).
        with torch.no_grad():
            # Iterate through the validation data loader.
            for inputs, labels in val_loader:
                # Move data to the device.
                inputs, labels = inputs.to(device), labels.to(device)
                # Perform a forward pass.
                outputs = model(inputs)
                # Compute the validation loss.
                loss = criterion(outputs, labels)
                # Accumulate the validation loss.
                val_loss += loss.item() * inputs.size(0)
                # Count correct predictions for validation.
                val_correct += (outputs.argmax(1) == labels).sum().item()
                # Add the number of samples to the total validation count.
                val_total += labels.size(0)
        # Calculate and store the average validation loss for the epoch.
        val_losses.append(val_loss / val_total)
        # Print the epoch's training and validation metrics.
        print(f"[{epoch+1}/{config['num_epochs']}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Train Acc: {accs[-1]:.4f} | Val Acc: {val_correct/val_total:.4f}")

    # --- Save Final Model ---
    # Create the directory for the final model save path if it doesn't exist.
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    # Save the state dictionary of the trained model to the specified path.
    torch.save(model.state_dict(), config["save_path"])
    # Print a confirmation message.
    print(f"✅ Model saved to {config['save_path']}")

    # --- Evaluation on Validation Set ---
    # Initialize lists to store true and predicted labels for detailed evaluation metrics.
    y_true, y_pred = [], []
    # Set the model back to evaluation mode.
    model.eval()
    # Iterate through the validation data loader without gradients.
    for inputs, labels in val_loader:
        # Move data to the device.
        inputs, labels = inputs.to(device), labels.to(device)
        # Perform a forward pass to get outputs.
        outputs = model(inputs)
        # Extend the true labels list with the ground truth labels (move to CPU and convert to NumPy).
        y_true.extend(labels.cpu().numpy())
        # Extend the predicted labels list with the model's predictions (argmax gets the predicted class index),
        # move to CPU, and convert to NumPy.
        y_pred.extend(outputs.argmax(1).cpu().numpy())

    # --- Save Evaluation Metrics ---
    # Open the metrics YAML file for writing.
    with open(os.path.join(results_path, "metrics.yaml"), "w") as f:
        # Generate a classification report (precision, recall, f1-score, support).
        # target_names provides labels for the classes.
        # output_dict=True returns the report as a dictionary.
        # zero_division=0 handles cases where a class has no samples in true or predicted labels.
        report = classification_report(
            y_true,
            y_pred,
            target_names=val_dataset.classes,
            output_dict=True,
            zero_division=0
        )
        # Dump the classification report dictionary to the YAML file.
        yaml.dump(report, f)

    # --- Plot and Save Loss Curves ---
    plt.figure() # Create a new figure.
    plt.plot(train_losses, label='Train Loss') # Plot the training loss over epochs.
    plt.plot(val_losses, label='Val Loss')     # Plot the validation loss over epochs.
    plt.title("Loss Curves") # Set the title of the plot.
    plt.legend() # Display the legend for the plot.
    plt.savefig(os.path.join(results_path, "loss_curve.png")) # Save the plot to a file.
    plt.close() # Close the figure to free memory.

    # --- Plot and Save Accuracy Curve ---
    plt.figure() # Create a new figure.
    plt.plot(accs, label='Train Accuracy') # Plot the training accuracy over epochs.
    plt.title("Accuracy Curve") # Set the title of the plot.
    plt.legend() # Display the legend.
    plt.savefig(os.path.join(results_path, "accuracy_curve.png")) # Save the plot.
    plt.close() # Close the figure.

    # --- Plot and Save Confusion Matrix ---
    # Generate the confusion matrix.
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8)) # Create a new figure with a specified size.
    plt.imshow(conf_matrix, cmap='Blues') # Display the confusion matrix as an image with a blue colormap.
    plt.title("Confusion Matrix") # Set the title.
    plt.colorbar() # Add a color bar to the plot.
    # Set the tick locations and labels for the x and y axes using class names.
    ticks = np.arange(len(val_dataset.classes))
    plt.xticks(ticks, val_dataset.classes, rotation=90) # Rotate x-axis labels for readability.
    plt.yticks(ticks, val_dataset.classes)
    # Determine the threshold for text color based on the maximum value in the matrix.
    thresh = conf_matrix.max() / 2.
    # Add text labels (counts) to each cell of the confusion matrix.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            # Get the cell value, handle if it's a single-element tensor.
            value = conf_matrix[i, j].item() if hasattr(conf_matrix[i, j], 'item') else conf_matrix[i, j]
            # Add text at the cell's center, with color determined by the threshold.
            plt.text(j, i, f'{value}',
                     horizontalalignment="center",
                     color="white" if value > thresh else "black")
    plt.tight_layout() # Adjust layout to prevent labels from overlapping.
    plt.savefig(os.path.join(results_path, "confusion_matrix.png")) # Save the plot.
    plt.close() # Close the figure.

    # --- Cleanup ---
    # Signal the GPU logging thread to stop.
    stop_event.set()
    # Wait for the GPU logging thread to finish.
    thread.join()
    # If the CodeCarbon tracker was started, stop it.
    if tracker:
        tracker.stop()

# --- Main Execution Block ---
# This block runs only when the script is executed directly (not imported as a module).
if __name__ == "__main__":
    # --- Argument Parsing ---
    # Create an argument parser object.
    parser = argparse.ArgumentParser()
    # Add an argument for the configuration file path.
    # '--config': The name of the argument.
    # type=str: Expected type is string.
    # required=True: This argument must be provided.
    # help: Description for the argument in the help message.
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    # Parse the command-line arguments.
    args = parser.parse_args()
    # --- Load Configuration and Start Training ---
    # Open the specified configuration file in read mode.
    with open(args.config, "r") as f:
        # Load the YAML configuration from the file.
        config = yaml.safe_load(f)
    # Call the main training function with the loaded configuration.
    train_model(config)
