"""
This script is designed to train and evaluate a computer program, known as a model, that can understand and classify pieces of text. Think of it like teaching a virtual assistant to tell whether movie reviews are positive or negative.

Text Classification
This script is designed to train a model for the purpose of text classification. Specifically, it aims to discern whether movie reviews carry a positive or negative sentiment.

Model Training
During the training phase, the model is "taught" through exposure to a dataset of known reviews. This process involves iterative adjustments to the model's parameters to enhance its predictive accuracy based on the feedback received.

Evaluation
Post-training, the script undertakes an evaluation phase where the model is tested against new, unseen movie reviews. This step is analogous to administering a final exam to gauge the model's learning efficacy.

Accuracy Tracking
The script meticulously calculates and logs the model's accuracy - defined as the percentage of reviews correctly identified. This metric serves as a key indicator of the model's performance over time.

Performance Logging
Detailed logs concerning the model's performance are maintained throughout the training process. These logs capture critical insights on improvements and adjustments made to the model.

Visual Representation
To visually depict the model's learning trajectory, the script generates a plot. This graphical representation showcases the evolution of the model's ability to accurately classify reviews as it undergoes training.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
import logging
import matplotlib.pyplot as plt
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO)

def evaluate_model(model, dataset, tokenizer, device):
    """
    Evaluate the model's performance on a given dataset.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - dataset (datasets.Dataset): The dataset to evaluate the model on.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for preprocessing text data.
    - device (torch.device): The device to run the evaluation on (e.g., 'cuda', 'cpu').

    Returns:
    - accuracy (float): The accuracy of the model on the given dataset.
    """

    logging.info("Starting model evaluation...")
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8)
    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            true_labels.extend(batch['label'].numpy())
            predictions.extend(preds)

    accuracy = accuracy_score(true_labels, predictions)
    logging.info("Model evaluation completed.")
    return accuracy

def train_model(model, train_loader, tokenizer, device):
    """
    Train the model on a dataset.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - train_loader (torch.utils.data.DataLoader): The DataLoader providing the training dataset.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer for preprocessing the text data.
    - device (torch.device): The device to perform the training on (e.g., 'cuda', 'cpu').

    This function performs training over a fixed number of epochs, logging the loss every 10 batches
    and the average loss at the end of each epoch. It also generates a plot of the average training loss
    per epoch after the training is complete.
    """

    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epoch_losses = []

    for epoch in range(10):
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = batch['label'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item()}")

        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
        epoch_losses.append(avg_loss)

    # Plotting the average loss per epoch after training
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), epoch_losses, label='Average Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Training Loss per Epoch')
    plt.legend()
    plt.savefig('average_training_loss_per_epoch.png')
    plt.close()

def main():
    """
    The main function to execute the training and evaluation of the model.

    This function sets up the device for training (GPU or CPU), loads the model and tokenizer,
    prepares the training and evaluation datasets, and invokes the training and evaluation processes.
    It logs the evaluation accuracy at the end of the training.

    This function is designed to run the training and evaluation on a single GPU or CPU without
    distributed training, simplifying the process for educational purposes or small-scale experiments.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)

    logging.info("Training on device: {}".format(device))

    train_dataset = load_from_disk("data/train_dataset")
    eval_dataset = load_from_disk("data/test_dataset")

    train_loader = DataLoader(train_dataset, batch_size=8)

    train_model(model, train_loader, tokenizer, device)

    eval_accuracy = evaluate_model(model, eval_dataset, tokenizer, device)
    logging.info(f"Evaluation Accuracy: {eval_accuracy}")

if __name__ == "__main__":
    main()
