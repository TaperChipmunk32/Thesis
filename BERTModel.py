import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pandas as pd

'''
BibleDataset class that takes a DataFrame with verse text and labels, a tokenizer, and a maximum sequence length as input.
Used to create a PyTorch Dataset for training and validation data.
'''
class BibleDataset(Dataset):
    '''
    Parameters:
        data (DataFrame): DataFrame with verse text and labels
        tokenizer (BertTokenizer): Tokenizer to convert verse text to input IDs and attention mask
        max_len (int): Maximum sequence length
    '''
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    '''
    Get a verse text and label from the DataFrame and encode the verse text using the tokenizer   
    Returns:
        dict: Dictionary with 'input_ids', 'attention_mask', and 'labels' keys
    '''
    def __getitem__(self, idx):
        verse = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            verse,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }

'''
BibleClassifier class that defines a BERT-based classifier model for Bible verse classification.
'''
class BibleClassifier(torch.nn.Module):
    def __init__(self):
        super(BibleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs
    
def BERT_EXPERIMENT(df):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Split the data into training and validation sets
    train_text, val_text, train_labels, val_labels = train_test_split(df['verse'], df['label'], test_size=0.2, stratify=df['label'], random_state=0)

    train_data = pd.DataFrame({'verse': train_text, 'label': train_labels})
    val_data = pd.DataFrame({'verse': val_text, 'label': val_labels})

    train_dataset = BibleDataset(train_data, tokenizer, 512)
    val_dataset = BibleDataset(val_data, tokenizer, 512)

    train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=8)

    model = BibleClassifier()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Test cases
    assert len(train_dataset) > 0, "Training dataset is empty"
    assert len(val_dataset) > 0, "Validation dataset is empty"

    # Test a batch from the training data loader
    train_batch = next(iter(train_data_loader))
    assert train_batch['input_ids'].shape[1] == 512, "Input IDs have incorrect shape"
    assert train_batch['attention_mask'].shape[1] == 512, "Attention mask has incorrect shape"
    assert train_batch['labels'].shape[0] == train_batch['input_ids'].shape[0], "Labels have incorrect shape"

    # Test a batch from the validation data loader
    val_batch = next(iter(val_data_loader))
    assert val_batch['input_ids'].shape[1] == 512, "Input IDs have incorrect shape"
    assert val_batch['attention_mask'].shape[1] == 512, "Attention mask has incorrect shape"
    assert val_batch['labels'].shape[0] == val_batch['input_ids'].shape[0], "Labels have incorrect shape"

    # Test the model's forward pass
    input_ids = train_batch['input_ids'].to(device)
    attention_mask = train_batch['attention_mask'].to(device)
    labels = train_batch['labels'].to(device)
    outputs = model(input_ids, attention_mask)
    assert outputs.shape[0] == input_ids.shape[0], "Model output has incorrect shape"
    assert outputs.shape[1] == 4, "Model output has incorrect number of classes"

    # Train the model on a small portion of the dataset to ensure it runs without errors
    train_subset = torch.utils.data.Subset(train_dataset, range(100))
    train_subset_loader = DataLoader(train_subset, batch_size=8, shuffle=True)

    print('Training the model on a subset of the data...')
    for epoch in range(2):
        model.train()
        total_loss = 0
        for batch in train_subset_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
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
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
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
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
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
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(val_data)
        print(f'Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}')