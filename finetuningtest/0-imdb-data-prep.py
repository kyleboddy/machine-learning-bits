"""
This script is designed to prepare a dataset for a machine learning project that aims to understand people's opinions about movies based on their reviews. Imagine you're planning to watch a movie and want to know if it's good or bad. You might read several reviews to decide. Similarly, this script helps in collecting and organizing a large number of movie reviews so that a computer program can learn to tell if a review suggests the movie is good or bad. 

Here's a simple breakdown of what it does:
- It fetches a collection of movie reviews from a large database.
- It then selects a specific number of these reviews to be used for training and testing the computer program. Think of this as picking out a set of reviews to teach the program what good and bad reviews look like.
- Finally, it saves these selected reviews in a structured format, making it easier for the computer program to learn from them.

This preparation step is crucial for ensuring that the computer program has a high-quality set of examples to learn from, which helps it make accurate predictions about new movie reviews it hasn't seen before.
"""

from datasets import load_dataset
import os

def prepare_imdb_dataset(train_size=5000, test_size=1000):
    """
    Prepares the IMDB dataset for training and testing by selecting a specified number of samples.

    This function loads the IMDB dataset, shuffles it, and then selects a subset for training and testing based on the specified sizes. It ensures that the requested sizes do not exceed the actual sizes of the dataset. The selected subsets are then saved to disk for future use.

    Parameters:
    - train_size (int): The desired size of the training dataset. Defaults to 5000.
    - test_size (int): The desired size of the testing dataset. Defaults to 1000.

    Outputs:
    - Saves the prepared training and testing datasets to the 'data' directory.
    - Prints the sizes of the prepared training and testing datasets.
    """
    dataset = load_dataset("imdb")
    # Ensure the requested sizes do not exceed dataset's actual sizes
    max_train_size = len(dataset["train"])
    max_test_size = len(dataset["test"])
    # Select the smaller of requested size or maximum available size
    train_dataset_size = min(train_size, max_train_size)
    test_dataset_size = min(test_size, max_test_size)

    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_dataset_size))  # Increased size for training
    test_dataset = dataset["test"].shuffle(seed=42).select(range(test_dataset_size))  # Increased size for testing

    # Saving datasets for future use
    os.makedirs("data", exist_ok=True)
    train_dataset.save_to_disk("data/train_dataset")
    test_dataset.save_to_disk("data/test_dataset")
    print(f"Datasets prepared and saved. Train size: {train_dataset_size}, Test size: {test_dataset_size}.")

if __name__ == "__main__":
    prepare_imdb_dataset(train_size=10000, test_size=5000)