"""
This script is part of a project aimed at teaching a computer to understand and categorize text, specifically movie reviews, into positive or negative sentiments. Imagine you're trying to decide whether to watch a movie and you're reading reviews to see if people liked it or not. This program aims to do something similar by learning from a large number of movie reviews.

Here's a simple breakdown of what the script does:
- It sets up a special environment that allows multiple computers (or parts of a computer) to work together, speeding up the learning process.
- It prepares the data: the script loads movie reviews that have been previously organized and saved.
- It trains the model: the script teaches the computer program by showing it examples of positive and negative reviews. It adjusts the program's internal settings to improve its ability to categorize new reviews it hasn't seen before.
- It evaluates the model: after training, the script checks how well the program can categorize new reviews, providing a measure of its accuracy.
- It repeats the training with different settings to find the best configuration for the program.

This script is designed for use by researchers or developers in the field of machine learning, specifically those working on natural language processing tasks."
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
import logging
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Initialize logging to both console and file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = f'training_log_{timestamp}.txt'
file_handler = logging.FileHandler(log_filename, mode='w')
logger.addHandler(file_handler)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_predictions(model, dataset, tokenizer, device, num_samples=10):  # Default set to 10
    model.eval()
    dataloader = DataLoader(dataset, batch_size=num_samples)
    predictions = []
    true_labels = []
    texts = []

    with torch.no_grad():
        batch = next(iter(dataloader))
        inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt").to(device)
        labels = batch['label'].to(device)
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        texts.extend(batch['text'])
        true_labels.extend(batch['label'].numpy())
        predictions.extend(preds)

    return texts, true_labels, predictions


def evaluate_model(model, dataset, tokenizer, device):
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

def train_model(model, train_loader, tokenizer, device, rank, num_epochs, lr):
    optimizer = AdamW(model.parameters(), lr=lr)
    epoch_losses = []

    for epoch in range(num_epochs):
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

            if num_batches % 10 == 0 and rank == 0:
                logging.info(f"Rank {rank}, Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item()}")

        avg_loss = total_loss / num_batches
        if rank == 0:
            logging.info(f"Rank {rank}, Epoch {epoch+1}, Average Loss: {avg_loss}")
            epoch_losses.append(avg_loss)

    if rank == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), epoch_losses, label='Average Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title(f'Average Training Loss per Epoch with LR={lr}')
        plt.legend()
        plt.savefig(f'loss_plot_lr_{lr}.png')
        plt.close()

def iterative_training(rank, world_size, learning_rates, num_epochs=10):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = load_from_disk("data/train_dataset")
    eval_dataset = load_from_disk("data/test_dataset")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)

    for lr in learning_rates:
        # Reload the model for each learning rate to reset its state
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
        model = DDP(model, device_ids=[rank])

        if rank == 0:
            pre_eval_accuracy = evaluate_model(model.module, eval_dataset, tokenizer, device)
            logging.info(f"Pre-training Evaluation Accuracy with LR={lr}: {pre_eval_accuracy}")

        logging.info(f"Training with Learning Rate: {lr}")
        train_model(model, train_loader, tokenizer, device, rank, num_epochs, lr)

        if rank == 0:
            post_eval_accuracy = evaluate_model(model.module, eval_dataset, tokenizer, device)
            logging.info(f"Post-training Evaluation Accuracy with LR={lr}: {post_eval_accuracy}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    learning_rates = [1e-5, 5e-5, 1e-4]  # Define your range of learning rates here
    logging.info(f"Number of GPUs available: {world_size}")
    torch.multiprocessing.spawn(iterative_training, args=(world_size, learning_rates), nprocs=world_size, join=True)
