import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch.nn as nn
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BibleDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len
        self.vocab = self.build_vocab(data['verse'])
        self.vocab_size = len(self.vocab)

    def build_vocab(self, verses):
        vocab = set()
        for verse in verses:
            for word in verse.split():
                vocab.add(word)
        vocab = {word: i + 2 for i, word in enumerate(vocab)}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    def encode(self, verse):
        encoding = [self.vocab.get(word, self.vocab['<UNK>']) for word in verse.split()]
        encoding = encoding[:self.max_len]
        encoding += [self.vocab['<PAD>']] * (self.max_len - len(encoding))
        return torch.tensor(encoding)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        verse = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.encode(verse)

        return {
            'input_ids': encoding,
            'labels': torch.tensor(label, dtype=torch.long),
        }

class BibleClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout, vocab_size):
        super(BibleClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, input_ids):
        embedding_out = self.embedding(input_ids)
        packed_out = pack_padded_sequence(embedding_out, torch.count_nonzero(input_ids, dim=1).cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_out)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = torch.cat((lstm_out[:, -1, :self.lstm.hidden_size], lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        dropout_out = self.dropout(lstm_out)
        outputs = self.fc(dropout_out)
        return outputs


def LSTM_EXPERIMENT():    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_text, val_text, train_labels, val_labels = train_test_split(df['verse'], df['label'], test_size=0.2, stratify=df['label'], random_state=0)

    train_data = pd.DataFrame({'verse': train_text, 'label': train_labels})
    val_data = pd.DataFrame({'verse': val_text, 'label': val_labels})

    train_dataset = BibleDataset(train_data, 512)
    val_dataset = BibleDataset(val_data, 512)

    train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=8)

    model = BibleClassifier(embedding_dim=128, hidden_dim=128, output_dim=4, dropout=0.1, vocab_size=train_dataset.vocab_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Test cases
    assert len(train_dataset) > 0, "Training dataset is empty"
    assert len(val_dataset) > 0, "Validation dataset is empty"

    # Test a batch from the training data loader
    train_batch = next(iter(train_data_loader))
    assert train_batch['input_ids'].shape[1] == 512, "Input IDs have incorrect shape"
    assert train_batch['labels'].shape[0] == train_batch['input_ids'].shape[0], "Labels have incorrect shape"

    # Test a batch from the validation data loader
    val_batch = next(iter(val_data_loader))
    assert val_batch['input_ids'].shape[1] == 512, "Input IDs have incorrect shape"
    assert val_batch['labels'].shape[0] == val_batch['input_ids'].shape[0], "Labels have incorrect shape"

    # Test the model's forward pass
    input_ids = train_batch['input_ids'].to(device)
    labels = train_batch['labels'].to(device)
    outputs = model(input_ids)
    assert outputs.shape[0] == input_ids.shape[0], "Model output has incorrect shape"
    assert outputs.shape[1] == 4, "Model output has incorrect number of classes"

    # Train the model on a small portion of the dataset
    train_subset = torch.utils.data.Subset(train_dataset, range(100))
    train_subset_loader = DataLoader(train_subset, batch_size=8, shuffle=True)

    print('Training the model on a subset of the data...')
    for epoch in range(2):
        model.train()
        total_loss = 0
        for batch in train_subset_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_subset_loader)}')

        model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in val_data_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(val_data)
        print(f'Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}')

    # If all test cases pass, train the model on the full dataset
    print('Training the model on the full dataset...')
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data_loader)}')

        model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in val_data_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(val_data)
        print(f'Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}')