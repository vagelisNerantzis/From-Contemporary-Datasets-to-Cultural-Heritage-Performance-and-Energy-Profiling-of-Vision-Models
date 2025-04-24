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

# Path to the codecarbon lock file. This is used by codecarbon to prevent
# multiple instances from running simultaneously. We define it here to be able
# to manage it if necessary.
CODECARBON_LOCK_FILE = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", ".codecarbon.lock")

class FocalLoss(nn.Module):
    """
    Implements the Focal Loss function for classification tasks.
    Focal Loss is designed to address class imbalance by down-weighting
    the loss for well-classified examples and focusing on hard, misclassified examples.

    Args:
        gamma (float): Focusing parameter. Higher values reduce the relative loss
                       for well-classified examples.
        alpha (float): Alpha parameter, a weighting factor for each class.
                       Used to balance the importance of different classes.
    """
    def __init__(self, gamma=2, alpha=1):
        # Call the constructor of the parent class (nn.Module)
        super(FocalLoss, self).__init__()
        # Store the gamma parameter
        self.gamma = gamma
        # Store the alpha parameter
        self.alpha = alpha
        # Initialize the standard Cross-Entropy Loss.
        # reduction='none' ensures that the loss is computed per element in the batch
        # rather than being averaged or summed immediately. This is needed
        # to apply the focal loss weighting.
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
        # Compute the standard Cross-Entropy Loss for each element in the batch.
        ce_loss = self.ce(inputs, targets)
        # Calculate the probability of the predicted class using the negative
        # exponent of the cross-entropy loss. pt = exp(-CE_loss)
        pt = torch.exp(-ce_loss)
        # Compute the Focal Loss based on the formula:
        # alpha * (1 - pt)^gamma * CE_loss
        # (1 - pt)^gamma is the modulating factor that reduces the loss for
        # easy examples (where pt is high) and increases it for hard examples
        # (where pt is low).
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # Return the mean of the focal loss over the batch.
        return focal_loss.mean()

def get_sampler_targets(dataset):
    """
    Creates a WeightedRandomSampler to handle class imbalance in the dataset.
    It assigns higher weights to samples from minority classes and lower weights
    to samples from majority classes, ensuring that each class is sampled
    approximately equally often during training.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to create the sampler for.
                                           Assumes the dataset has a .samples attribute
                                           where each sample is a tuple (image_path, label).

    Returns:
        torch.utils.data.WeightedRandomSampler: The sampler object.
    """
    # Extract the target labels for each sample in the dataset.
    targets = [s[1] for s in dataset.samples]
    # Count the occurrences of each class label.
    class_counts = Counter(targets)
    # Calculate the weight for each sample. The weight is inversely proportional
    # to the frequency of its class. This means samples from less frequent
    # classes get higher weights.
    weights = [1.0 / class_counts[t] for t in targets]
    # Create a WeightedRandomSampler.
    # weights: The list of weights for each sample.
    # num_samples: The total number of samples to draw (usually the size of the dataset).
    # replacement=True: Samples are drawn with replacement, meaning a sample can be
    #                   selected multiple times in an epoch. This is typical for
    #                   weighted sampling to oversample minority classes.
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def clean_codecarbon_lock():
    """
    Deletes the codecarbon lock file if it exists.
    This is sometimes necessary if a previous run of codecarbon crashed
    and left the lock file behind, preventing new instances from starting.

    Returns:
        bool: True if the lock file was successfully removed or didn't exist,
              False if an error occurred during removal.
    """
    # Check if the codecarbon lock file exists at the predefined path.
    if os.path.exists(CODECARBON_LOCK_FILE):
        try:
            # Attempt to remove the lock file.
            os.remove(CODECARBON_LOCK_FILE)
            # Print a success message if the file is removed.
            print(f"✅ Removed stale codecarbon lock file: {CODECARBON_LOCK_FILE}")
            return True
        except Exception as e:
            # Print a warning message if an error occurs during removal.
            print(f"⚠️ Failed to remove codecarbon lock file: {str(e)}")
            return False
    # Return True if the lock file did not exist in the first place.
    return True

def log_gpu_power(log_file, stop_event):
    """
    Continuously logs the GPU power consumption to a specified file.
    This function runs in a separate thread during training to monitor
    energy usage in real-time.

    Args:
        log_file (str): The path to the file where GPU power data will be logged.
        stop_event (threading.Event): An event object used to signal the thread
                                      to stop logging.
    """
    # Loop continuously until the stop_event is set.
    while not stop_event.is_set():
        try:
            # Run the nvidia-smi command to query the GPU power draw.
            # '--query-gpu=power.draw': Specifies the data to query (power draw).
            # '--format=csv,noheader': Formats the output as CSV with no header row.
            # capture_output=True: Captures the standard output and standard error.
            # text=True: Decodes the output as text.
            # check=True: Raises a CalledProcessError if the command returns a non-zero exit code.
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader'], capture_output=True, text=True, check=True)
            # Extract the power usage string from the standard output and remove leading/trailing whitespace.
            power_usage = result.stdout.strip()
            # Get the current timestamp in a specific format.
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            # Open the log file in append mode ('a') and write the timestamp and power usage.
            with open(log_file, 'a') as f:
                f.write(f"{timestamp},{power_usage}\n")
        except subprocess.CalledProcessError as e:
            # Handle errors specifically related to running the nvidia-smi command.
            print(f"⚠️ Error running nvidia-smi: {e}")
            # Break the loop if there's an error with the command.
            break
        except Exception as e:
            # Handle any other unexpected errors during logging.
            print(f"⚠️ An error occurred during GPU power logging: {e}")
            # Break the loop if any other error occurs.
            break
        # Wait for 5 seconds before the next query to avoid excessive logging and resource usage.
        time.sleep(5)

def train_model(config):
    """
    Orchestrates the entire training process for the ResNet50 model.
    This function handles dataset loading, model initialization, training loop,
    evaluation, saving results, and energy consumption tracking.

    Args:
        config (dict): A dictionary containing training configuration parameters.
                       Expected keys include:
                       - 'results_path': Directory to save results (models, metrics, plots).
                       - 'data_dir': Directory containing 'train' and 'val' subdirectories with image data.
                       - 'batch_size': Training and validation batch size.
                       - 'num_epochs': Number of training epochs.
                       - 'image_size': The size to resize input images to (image_size x image_size).
                       - 'learning_rate': Learning rate for the optimizer.
                       - 'num_workers': Number of worker processes for data loading.
                       - 'save_path': Path to save the final trained model.
                       - 'loss_function': The loss function to use ('focal' or 'crossentropy').
                       - 'pretrained': Boolean indicating whether to use a pre-trained ResNet50.
    """
    # Get the path to the results directory from the configuration.
    results_path = config['results_path']
    # Check if the results directory exists.
    if not os.path.exists(results_path):
        # If it doesn't exist, print a message and create the directory.
        print(f"Creating results directory: {results_path}")
        os.makedirs(results_path, exist_ok=True) # exist_ok=True prevents an error if the directory already exists (e.g., due to a race condition).
    else:
        # If the directory exists, print a message.
        print(f"Results directory exists: {results_path}")

    # --- Check Write Permissions ---
    # Verify that the script has write permissions in the results directory.
    try:
        # Create a temporary file path within the results directory.
        test_file_path = os.path.join(results_path, "test_write_permissions.txt")
        # Attempt to open and write to the temporary file.
        with open(test_file_path, 'w') as f:
            f.write("Test write permissions")
        # If writing is successful, remove the temporary file.
        os.remove(test_file_path)
        # Print a success message.
        print(f"✅ Write permissions confirmed for: {results_path}")
    except Exception as e:
        # If an error occurs during writing or removing, it indicates a permission issue.
        print(f"❌ No write permissions for {results_path}: {str(e)}")
        # Guide the user on how to resolve the issue.
        print("Please check directory permissions or specify a different path.")
        # Exit the script as it cannot proceed without write permissions.
        sys.exit(1)

    # --- CodeCarbon Tracker Initialization ---
    # Clean up the codecarbon lock file before initializing the tracker.
    lock_removed = clean_codecarbon_lock()

    # Initialize the EmissionsTracker variable to None.
    tracker = None
    try:
        # Print a message indicating the start of EmissionsTracker initialization.
        print("Initializing EmissionsTracker...")
        # Define the path for the emissions output file.
        emissions_file = os.path.join(results_path, "emissions.csv")
        # Print the path where emissions data will be saved.
        print(f"Emissions will be saved to: {emissions_file}")

        # Only initialize and start the tracker if the lock file was successfully removed or didn't exist.
        if lock_removed:
            # Create an instance of EmissionsTracker.
            tracker = EmissionsTracker(
                output_dir=results_path, # Directory to save emissions data.
                output_file="emissions.csv", # Name of the emissions file.
                measure_power_secs=15, # Frequency (in seconds) to measure power consumption.
                log_level="warning", # Set logging level to warning to reduce verbosity.
                save_to_file=True, # Enable saving emissions data to a file.
                allow_multiple_runs=True # Allow tracking multiple runs in the same directory.
            )
            # Print a message indicating the start of tracking.
            print("Starting EmissionsTracker...")
            # Start the emissions tracker. This begins monitoring energy consumption.
            tracker.start()
            # Print a success message.
            print("✅ EmissionsTracker started successfully")
        else:
            # If the lock file could not be removed, print a warning and disable energy tracking.
            print("⚠️ Unable to remove codecarbon lock file. Energy tracking will be disabled.")
    except Exception as e:
        # If any error occurs during tracker initialization or start, handle it.
        print(f"\n⚠️ CodeCarbon Tracker error: {str(e)}")
        # Ask the user if they want to continue training without energy tracking.
        user_input = input("❓ Do you want to continue training without energy tracking? [y/n]: ")
        # If the user does not input 'y' (case-insensitive), exit the script.
        if user_input.lower() != 'y':
            print("⛔ Exiting training.")
            sys.exit(1)
        # If the user inputs 'y', set the tracker to None and continue without tracking.
        tracker = None
        print("Continuing without energy tracking")

    # --- Log Training Start Information ---
    # Record the start time of training and other relevant information in a file.
    with open(os.path.join(results_path, "training_start.txt"), "w") as f:
        # Write the start timestamp.
        f.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        # Write the device being used for training (CUDA if available, otherwise CPU).
        f.write(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        # Indicate whether the EmissionsTracker is active.
        f.write(f"EmissionsTracker active: {tracker is not None}\n")

    # --- Start GPU Power Logging Thread ---
    # Initialize and start a separate thread to continuously log GPU power usage.
    gpu_log_file = os.path.join(results_path, "gpu_power.log") # Define the GPU power log file path.
    stop_gpu_log_event = threading.Event() # Create an event to signal the logging thread to stop.
    # Create a new thread targeting the log_gpu_power function with necessary arguments.
    gpu_log_thread = threading.Thread(target=log_gpu_power, args=(gpu_log_file, stop_gpu_log_event))
    # Start the GPU power logging thread.
    gpu_log_thread.start()
    # Print a message indicating where GPU power usage is being logged.
    print(f"📊 GPU power usage will be logged to: {gpu_log_file}")

    # --- Load Configuration Parameters ---
    # Extract training parameters from the loaded configuration dictionary.
    data_dir = config["data_dir"]           # Directory containing image datasets.
    batch_size = config["batch_size"]       # Number of samples per batch.
    num_epochs = config["num_epochs"]       # Total number of training epochs.
    image_size = config["image_size"]       # Target size for image resizing.
    learning_rate = config["learning_rate"] # Learning rate for the optimizer.
    num_workers = config["num_workers"]     # Number of subprocesses for data loading.
    save_path = config["save_path"]         # Path to save the final model state dictionary.
    loss_function = config["loss_function"] # Name of the loss function to use.

    # --- Data Transformations ---
    # Define image transformations to be applied to the dataset.
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), # Resize images to the specified size.
        transforms.ToTensor() # Convert images to PyTorch tensors (scales pixel values to [0, 1]).
    ])

    # --- Dataset Loading ---
    # Load the training and validation datasets using ImageFolder.
    print(f"Loading datasets from {data_dir}...")
    # Load the training dataset from the 'train' subdirectory.
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    # Load the validation dataset from the 'val' subdirectory.
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    # Print the number of images found in each dataset.
    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images")

    # --- Sampler for Class Imbalance ---
    # Get the weighted random sampler for the training dataset to handle class imbalance.
    sampler = get_sampler_targets(train_dataset)

    # --- Data Loaders ---
    # Create data loaders for the training and validation datasets.
    # train_loader: Uses the weighted sampler for balanced batch sampling.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    # val_loader: Uses standard shuffling (set to False as shuffling is not typically needed for validation)
    # and the specified batch size.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Model Initialization ---
    # Initialize the ResNet50 model.
    if config["pretrained"]:
        # If 'pretrained' is True in the config, load a pre-trained ResNet50 model
        # with ImageNet weights.
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        # If 'pretrained' is False, initialize a ResNet50 model with random weights.
        model = models.resnet50(weights=None)

    # --- Modify the Classifier Head ---
    # Get the number of input features for the final fully connected layer.
    num_ftrs = model.fc.in_features
    # Replace the original fully connected layer with a new one that has the
    # correct number of output features corresponding to the number of classes
    # in the training dataset.
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

    # --- Device Configuration ---
    # Determine the device to use for training (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Print the device being used.
    print(f"Using device: {device}")
    # Move the model to the selected device.
    model = model.to(device)

    # --- Loss Function Selection ---
    # Select the loss function based on the configuration.
    if loss_function == "focal":
        # If 'focal' is specified, use the custom FocalLoss.
        criterion = FocalLoss()
        print("Using Focal Loss")
    else:
        # Otherwise, default to Cross-Entropy Loss.
        criterion = nn.CrossEntropyLoss()
        print("Using Cross Entropy Loss")

    # --- Optimizer Configuration ---
    # Initialize the Adam optimizer with the model's parameters and the specified learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Lists to Store Metrics ---
    # Initialize lists to store training and validation losses and training accuracies per epoch.
    train_losses = []
    val_losses = []
    accs = []

    # --- Training Loop ---
    try:
        # Print a message indicating the start of the training process and the number of epochs.
        print(f"Starting training for {num_epochs} epochs...")
        # Iterate through each epoch.
        for epoch in range(num_epochs):
            # Set the model to training mode. This enables dropout and batch normalization updates.
            model.train()
            running_loss = 0.0 # Initialize the running loss for the current epoch.
            correct = 0 # Initialize the count of correctly predicted training samples.
            total = 0 # Initialize the total number of training samples processed in the current epoch.

            # Iterate through the training data loader to get batches of data.
            for inputs, labels in train_loader:
                # Move the input images and labels to the selected device (GPU or CPU).
                inputs, labels = inputs.to(device), labels.to(device)
                # Zero the gradients of the optimizer. This is crucial before performing a backward pass.
                optimizer.zero_grad()
                # Perform a forward pass: get model outputs (logits) for the current batch.
                outputs = model(inputs)
                # Compute the loss using the selected criterion (FocalLoss or CrossEntropyLoss).
                loss = criterion(outputs, labels)
                # Perform a backward pass: compute gradients of the loss with respect to model parameters.
                loss.backward()
                # Update model parameters based on the computed gradients.
                optimizer.step()
                # Accumulate the batch loss, weighted by the number of samples in the batch.
                running_loss += loss.item() * inputs.size(0)
                # Get the index of the class with the highest probability (predicted class).
                _, predicted = torch.max(outputs.data, 1)
                # Update the total number of samples processed.
                total += labels.size(0)
                # Update the count of correctly predicted samples.
                correct += (predicted == labels).sum().item()

            # Calculate the average training loss for the current epoch.
            train_loss = running_loss / total
            # Calculate the training accuracy for the current epoch.
            train_acc = correct / total
            # Append the training loss to the list.
            train_losses.append(train_loss)
            # Append the training accuracy to the list.
            accs.append(train_acc)

            # --- Validation Phase ---
            # Set the model to evaluation mode. This disables dropout and uses batch normalization statistics
            # collected during training.
            model.eval()
            val_loss = 0.0 # Initialize validation loss.
            val_total = 0 # Initialize total validation samples.
            val_correct = 0 # Initialize count of correctly predicted validation samples.

            # Disable gradient calculation during validation as it's not needed for inference.
            with torch.no_grad():
                # Iterate through the validation data loader.
                for inputs, labels in val_loader:
                    # Move inputs and labels to the device.
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Perform a forward pass.
                    outputs = model(inputs)
                    # Compute the validation loss.
                    loss = criterion(outputs, labels)
                    # Accumulate the batch validation loss.
                    val_loss += loss.item() * inputs.size(0)
                    # Get the predicted class indices.
                    _, predicted = torch.max(outputs, 1)
                    # Update total validation samples.
                    val_total += labels.size(0)
                    # Update count of correctly predicted validation samples.
                    val_correct += (predicted == labels).sum().item()

            # Calculate the average validation loss for the current epoch.
            val_losses.append(val_loss / val_total)
            # Calculate the validation accuracy for the current epoch.
            val_acc = val_correct / val_total
            # Print the training and validation metrics for the current epoch.
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_losses[-1]:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # --- Save Intermediate Checkpoints ---
            # Save the model's state dictionary periodically (every 5 epochs and at the last epoch).
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                # Define the path for the intermediate checkpoint file.
                temp_save_path = os.path.join(results_path, f"model_epoch_{epoch+1}.pth")
                # Save the model's state dictionary.
                torch.save(model.state_dict(), temp_save_path)
                # Print a message confirming the save.
                print(f"Saved intermediate model to {temp_save_path}")

        # --- Save Final Model ---
        # Create the directory for the final model save path if it doesn't exist.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the state dictionary of the final trained model.
        torch.save(model.state_dict(), save_path)
        # Print a message confirming the final model save.
        print(f"Saved final model to {save_path}")

        # --- Evaluation and Metric Calculation ---
        # Initialize lists to store true and predicted labels for evaluation.
        y_true = []
        y_pred = []
        # Set the model back to evaluation mode.
        model.eval()
        # Iterate through the validation data loader without gradient calculation.
        for inputs, labels in val_loader:
            # Move inputs and labels to the device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform a forward pass to get model outputs.
            outputs = model(inputs)
            # Get the predicted class indices.
            _, preds = torch.max(outputs, 1)
            # Extend the true labels list with the ground truth labels (move to CPU and convert to NumPy array).
            y_true.extend(labels.cpu().numpy())
            # Extend the predicted labels list with the model's predictions (move to CPU and convert to NumPy array).
            y_pred.extend(preds.cpu().numpy())

        # --- Generate Classification Report and Confusion Matrix ---
        # Generate a classification report (precision, recall, f1-score) using true and predicted labels.
        # target_names provides labels for each class in the report.
        # output_dict=True returns the report as a dictionary.
        class_report = classification_report(y_true, y_pred, target_names=val_dataset.classes, output_dict=True)
        # Generate a confusion matrix to visualize the performance of the classification model.
        conf_matrix = confusion_matrix(y_true, y_pred)

        # --- Save Evaluation Metrics and Plots ---
        print("Saving evaluation metrics and plots...")
        # Save the classification report to a YAML file.
        with open(os.path.join(results_path, "metrics.yaml"), "w") as f:
            yaml.dump(class_report, f)

        # Plot and save the loss curves (training and validation loss over epochs).
        plt.figure() # Create a new figure.
        plt.plot(train_losses, label='Train Loss') # Plot training loss.
        plt.plot(val_losses, label='Val Loss')     # Plot validation loss.
        plt.legend() # Display the legend.
        plt.title("Loss Curves") # Set the plot title.
        plt.savefig(os.path.join(results_path, "loss_curve.png")) # Save the plot as a PNG file.
        plt.close() # Close the figure to free up memory.

        # Plot and save the training accuracy curve.
        plt.figure() # Create a new figure.
        plt.plot(accs, label='Train Accuracy') # Plot training accuracy.
        plt.legend() # Display the legend.
        plt.title("Training Accuracy") # Set the plot title.
        plt.savefig(os.path.join(results_path, "accuracy_curve.png")) # Save the plot.
        plt.close() # Close the figure.

        # Plot and save the confusion matrix.
        plt.figure(figsize=(10, 8)) # Create a new figure with a specified size.
        plt.imshow(conf_matrix, cmap='Blues') # Display the confusion matrix as an image with a blue colormap.
        plt.title("Confusion Matrix") # Set the plot title.
        plt.colorbar() # Add a color bar to indicate the scale.
        # Set the tick marks and labels for the x and y axes to correspond to the class names.
        tick_marks = np.arange(len(val_dataset.classes))
        plt.xticks(tick_marks, labels=val_dataset.classes, rotation=90) # Rotate x-axis labels for readability.
        plt.yticks(tick_marks, labels=val_dataset.classes)

        # Add text labels (counts) inside the confusion matrix cells.
        thresh = conf_matrix.max() / 2. # Threshold to determine text color (white on dark background, black on light).
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                # Add text at the center of each cell with the count.
                # Color is set based on the cell value relative to the threshold.
                plt.text(j, i, f'{conf_matrix[i, j]}',
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")

        plt.tight_layout() # Adjust plot to prevent labels from overlapping.
        plt.ylabel('True label') # Set the y-axis label.
        plt.xlabel('Predicted label') # Set the x-axis label.
        plt.savefig(os.path.join(results_path, "confusion_matrix.png")) # Save the plot.
        plt.close() # Close the figure.

        # Print a message indicating that training is complete and results are saved.
        print("✅ Training complete. Metrics and model saved.")

    except Exception as e:
        # --- Error Handling During Training ---
        # If an error occurs during the training process, print an error message.
        print(f"❌ Error during training: {str(e)}")
        # Save a checkpoint of the model's current state when an error occurs.
        # This allows resuming training from the point of failure if possible.
        error_save_path = os.path.join(results_path, "model_error_checkpoint.pth")
        torch.save(model.state_dict(), error_save_path)
        print(f"Saved error checkpoint to {error_save_path}")
        # Re-raise the exception to allow further handling or to terminate the script.
        raise e
    finally:
        # --- Cleanup Actions (Executed regardless of whether training succeeded or failed) ---
        # Stop the GPU power logging thread.
        stop_gpu_log_event.set() # Set the event to signal the thread to stop.
        gpu_log_thread.join()    # Wait for the thread to finish execution.
        print(f"🛑 GPU power logging stopped.")

        # Stop the EmissionsTracker and save energy data if it was initialized and started.
        if tracker:
            print("Stopping EmissionsTracker and saving energy data...")
            try:
                # Stop the tracker and get the emissions data.
                emissions_data = tracker.stop()
                # Print the collected emissions data.
                print(f"Energy tracking complete. Emissions data: {emissions_data}")
                # Define the expected path for the emissions file.
                emissions_file = os.path.join(results_path, "emissions.csv")
                # Check if the emissions file was created by the tracker.
                if os.path.exists(emissions_file):
                    print(f"✅ Emissions data saved to {emissions_file}")
                else:
                    # If the file was not found, print a warning.
                    print(f"❌ Warning: Emissions file not found at {emissions_file}")
                    # If emissions data was returned by tracker.stop() but the file wasn't found,
                    # attempt to manually create the file and write the data.
                    if emissions_data:
                        with open(emissions_file, "w") as f:
                            f.write(f"timestamp,emissions_data\n{time.strftime('%Y-%m-%d %H:%M:%S')},{emissions_data}\n")
                        print(f"✅ Manually created emissions file at {emissions_file}")
            except Exception as e:
                # Handle any errors that occur while stopping the tracker.
                print(f"❌ Error stopping tracker: {str(e)}")

            # Clean up the codecarbon lock file after the tracker has stopped.
            clean_codecarbon_lock()

        # --- Log Training End Information ---
        # Record the end time of training and final metrics in a file.
        try:
            with open(os.path.join(results_path, "training_end.txt"), "w") as f:
                # Write the training end timestamp.
                f.write(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                # If training losses were recorded, write the final training loss.
                if train_losses:
                    f.write(f"Final train loss: {train_losses[-1]}\n")
                # If validation losses were recorded, write the final validation loss.
                if val_losses:
                    f.write(f"Final validation loss: {val_losses[-1]}\n")
                # If training accuracies were recorded, write the final training accuracy.
                if accs:
                    f.write(f"Final train accuracy: {accs[-1]}\n")
        except Exception as e:
            # Handle any errors that occur while saving training end statistics.
            print(f"❌ Error saving training end stats: {str(e)}")

# --- Main Execution Block ---
# This block is executed when the script is run directly.
if __name__ == "__main__":
    # --- Argument Parsing ---
    # Create an ArgumentParser object to handle command-line arguments.
    parser = argparse.ArgumentParser()
    # Add an argument for the configuration file path.
    # '--config': The name of the command-line argument.
    # type=str: The expected data type of the argument (string).
    # required=True: Indicates that this argument must be provided.
    # help: A brief description of the argument.
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    # Parse the command-line arguments.
    args = parser.parse_args()

    # --- Load Configuration and Start Training ---
    try:
        # Open the specified configuration file in read mode ('r').
        with open(args.config, "r") as f:
            # Load the YAML configuration from the file.
            config = yaml.safe_load(f)
        # Print a message confirming the configuration file has been loaded.
        print(f"Loaded configuration from {args.config}")
        # Call the train_model function with the loaded configuration.
        train_model(config)
    except Exception as e:
        # If any error occurs during configuration loading or training, print an error message.
        print(f"❌ Error: {str(e)}")
        # Exit the script with a non-zero status code to indicate an error.
        sys.exit(1)
