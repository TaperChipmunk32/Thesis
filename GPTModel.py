import torch
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random

class BibleDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

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

class BibleClassifier(torch.nn.Module):
    def __init__(self):
        super(BibleClassifier, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.gpt2.config.n_embd, 4)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs



def GPT_EXPERIMENT():    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
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

    # Train the model on a small portion of the dataset
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