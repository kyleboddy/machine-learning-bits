import torch
from transformers import BertForSequenceClassification

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
    print("Model loaded successfully on device:", device)

if __name__ == "__main__":
    test_model()
