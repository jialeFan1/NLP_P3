import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def prepare_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    sentences = []
    labels = []
    for line in lines:
        if line.strip() and line[0].isdigit():
            # Find the index of the first space to correctly split the label from the sentence
            first_space_index = line.find(' ')
            if first_space_index != -1:
                label = line[:first_space_index].strip()
                sentence = line[first_space_index+1:].strip()
                # Remove any colons or other non-digit characters from the label
                label = ''.join(filter(str.isdigit, label))
                try:
                    # Convert the cleaned label to an integer, adjusting for BERT label indexing
                    labels.append(int(label) - 1)  # BERT expects labels starting from 0
                    sentences.append(sentence)
                except ValueError:
                    print(f"Skipping line due to label parsing error: {line.strip()}")
                    continue
    return sentences, labels

def encode_data(tokenizer, sentences, labels, max_length):
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,  # Ensure that sentence length does not exceed max_length
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels
def train_and_save_model(data_path, model_save_path):
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    # Prepare data
    sentences, labels = prepare_data(data_path)
    input_ids, attention_masks, labels = encode_data(tokenizer, sentences, labels, max_length=64)

    # Split data
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=2018, test_size=0.1)

    # Data loaders
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=32)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    for epoch_i in range(0, epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch_i + 1}/{epochs} complete. Average loss: {total_loss / len(train_dataloader)}")

    # Save model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

# Define data paths and model save paths
datasets = {
    'conviction.txt': './BERT_WSD_model_conviction',
    'deed.txt': './BERT_WSD_model_deed',
    'diner.txt': './BERT_WSD_model_diner'
}

for data_path, model_save_path in datasets.items():
    print(f"Training model for {data_path}")
    train_and_save_model(data_path, model_save_path)
