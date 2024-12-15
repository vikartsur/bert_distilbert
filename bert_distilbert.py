import re
import random
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

file_path = "tweets.xlsx"
df = pd.read_excel(file_path)
tweets_exp= df.iloc[:, 2].astype(str).to_numpy()
tweets = np.tile(tweets_exp, 5)

#---
def clean_text(texts):
    cleaned_texts = []
    for text in texts:
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_texts.append(text)
    return cleaned_texts
tweets_v1 = clean_text(tweets)


#---
def normalize_text(texts):
    normalized_texts = []
    for text in texts:
        text = text.lower()
        normalized_texts.append(text)
    return normalized_texts
tweets_v2 = normalize_text(tweets_v1)


#---
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
def lemmatize_text(texts):
    lemmatizer = WordNetLemmatizer()
    lemmatized_texts = []
    for text in texts:
        words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        lemmatized_texts.append(' '.join(lemmatized_words))
    return lemmatized_texts
tweets_v3 = lemmatize_text(tweets_v2)


tweets = tweets_v3
#---
"""
books_keywords = ["книга", "манга", "роман", "твір", "автор", "персонажі", "сюжет", "літературний",
                  "бібліотека", "читати", "сторінки", "книга", "літературний жанр", "глава", "опис",
                  "мальовка"]
film_keywords = ["фільм", "кіно", "актор", "режисер", "сцена", "сюжет", "екшн", "драма", "комедія", "трейлер",
                  "прем'єра", "кінозал", "перегляд", "кінематограф", "сцена", "звукові ефекти", "кінокритик",
                 "серія", "серіал"]
"""
books_keywords = ["book", "manga", "novel", "work", "author", "characters", "plot", "literary",
                  "library", "read", "pages", "book", "literary genre", "chapter", "description",
                  "illustration"]
film_keywords =  ["film", "cinema", "actor", "director", "scene", "plot", "action", "drama", "comedy", "trailer",
                  "premiere", "cinema hall", "viewing", "cinematography", "scene", "sound effects", "film critic",
                  "series", "serial"]

labels = []
for tweet in tweets:
    if any(keyword in tweet for keyword in books_keywords):
        labels.append(0)
    elif any(keyword in tweet for keyword in film_keywords):
        labels.append(1)
    else:
        labels.append(2)

train_text, train_temp = train_test_split(tweets, test_size=0.4, random_state=42)
val_text, test_text = train_test_split(train_temp, test_size=0.5, random_state=42)
train_labels, labels_temp = train_test_split(labels, test_size=0.4, random_state=42)
val_labels, test_labels = train_test_split(labels_temp, test_size=0.5, random_state=42)

tokens_train = tokenizer.batch_encode_plus(
    train_text,
    max_length=32,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
tokens_val = tokenizer.batch_encode_plus(
    val_text,
    max_length=32,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
tokens_test = tokenizer.batch_encode_plus(
    test_text,
    max_length=32,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

train_seq = tokens_train['input_ids']
train_mask = tokens_train['attention_mask']
train_y = torch.tensor(train_labels)

val_seq = tokens_val['input_ids']
val_mask = tokens_val['attention_mask']
val_y = torch.tensor(val_labels)

test_seq = tokens_test['input_ids']
test_mask = tokens_test['attention_mask']
test_y = torch.tensor(test_labels)

batch_size = 8

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

class BERT_Arch(nn.Module):
    def __init__(self):
        super(BERT_Arch, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, sent_id, mask):
        outputs = self.bert(sent_id, attention_mask=mask)
        cls_hs = outputs.last_hidden_state[:, 0, :]
        x = self.fc1(cls_hs)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = BERT_Arch().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
class_counts = np.bincount(train_labels)
total_count = len(train_labels)
class_weights = [total_count / class_counts[i] if class_counts[i] != 0 else 0 for i in range(len(class_counts))]
class_weights = torch.tensor(class_weights).float().to(device)
cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

def train():
    model.train()
    total_loss = 0
    total_preds = []

    for batch in tqdm(train_dataloader):
        batch = [b.to(device) for b in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_preds.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

def evaluate():
    model.eval()
    total_loss = 0
    total_preds = []

    for batch in tqdm(val_dataloader):
        batch = [b.to(device) for b in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            total_preds.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


epochs = 1
best_valid_loss = float('inf')

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1} / {epochs}')
    train_loss, _ = train()
    valid_loss, _ = evaluate()
    print(f'Training loss: {train_loss:.3f}, Validation loss: {valid_loss:.3f}')

model.eval()
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    test_preds = preds.argmax(dim=1).cpu().numpy()

test_df = pd.DataFrame({"tweet": test_text, "target": test_labels, "pred": test_preds})
print(classification_report(test_df['target'], test_df['pred']))