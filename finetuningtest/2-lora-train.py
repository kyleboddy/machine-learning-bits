"""
This script is designed for a specific task in the field of artificial intelligence, focusing on natural language processing. It's part of a larger project aimed at teaching a computer to understand human language, specifically to classify text data - in this case, movie reviews - as either positive or negative. This process is known as sentiment analysis.

Here's a simple breakdown of what the script does:
- It sets up a special environment that allows the computer to work more efficiently, especially when dealing with large amounts of data or complex calculations.
- It prepares the data: Before the computer can learn from the movie reviews, the data needs to be organized and formatted in a way that the computer can process.
- It trains the model: Using the prepared data, the script teaches the computer by showing it examples of positive and negative reviews. The computer makes predictions, and based on the feedback, adjusts its parameters to improve its predictions.
- It evaluates the model: After training, the script tests the computer's ability to classify new reviews it hasn't seen before. This step checks how well the computer has learned to differentiate between positive and negative sentiments.
- It involves advanced techniques to improve the computer's learning capability, ensuring it can make accurate predictions.

This script is a crucial part of developing AI systems that can understand and interpret human language, with potential applications in areas such as customer service, content moderation, and more.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from datasets import load_from_disk
from sklearn.metrics import accuracy_score
import logging
import matplotlib.pyplot as plt
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

def train_model(model, train_loader, tokenizer, device, rank):
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epoch_losses = []  # Store average loss per epoch for plotting

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

            # Log loss every 10 batches
            if num_batches % 10 == 0 and rank == 0:
                print(f"Rank {rank}, Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item()}")

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        if rank == 0:
            print(f"Rank {rank}, Epoch {epoch+1}, Average Loss: {avg_loss}")
            epoch_losses.append(avg_loss)  # Append average loss for this epoch to the list

    # Plotting the average loss per epoch after training
    if rank == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), epoch_losses, label='Average Training Loss per Epoch')  # X-axis: Epoch numbers
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Average Training Loss per Epoch')
        plt.legend()
        plt.savefig('average_training_loss_per_epoch.png')
        plt.close()


def main(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
    model = DDP(model, device_ids=[rank])

    logging.info(f"Training on GPU: {rank}")

    train_dataset = load_from_disk("data/train_dataset")
    eval_dataset = load_from_disk("data/test_dataset")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler)

    train_model(model, train_loader, tokenizer, device, rank)

    if rank == 0:
        eval_accuracy = evaluate_model(model.module, eval_dataset, tokenizer, device)
        logging.info(f"Evaluation Accuracy: {eval_accuracy}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    logging.info(f"Number of GPUs available: {world_size}")
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)