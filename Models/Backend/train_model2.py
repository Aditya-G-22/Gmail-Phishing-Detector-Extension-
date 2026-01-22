#Imports
#Configuration
#Load Data
#Train-Test split
#Tokenization
#Train Model
    #torch dataset
    #Model
    #device setup
    #Data Loaders
    #Optimizer
    #Training Loop
#Evaluate Model
#Calibration
#Test Evaluation
#Saving

#--------Imports----------
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import json



#--------Configuration--------

data_path = "email_phishing_nlp_dataset.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5


#--------Load Data---------------

df = pd.read_csv(data_path)

#---------Train Test split------------

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify = df["label"] 
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    train_size = 0.5,
    random_state = RANDOM_STATE,
    stratify = temp_labels
)

#---------------tokenization-----------------------

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(texts) :
    return tokenizer(
        texts,
        padding = True,
        truncation = True,
        max_length = MAX_LENGTH,
        return_tensors="pt"
    )

#------------Model trainig----------------------------------

                #----------------torch dataset------------------------

class EmailDataset(torch.utils.data.Dataset) :
    def __init__(self, texts, labels) :
        self.encodings = tokenize(texts)
        self.labels = torch.tensor(labels)

    def __len__(self) :
        return len(self.labels)
    
    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item["labels"] = self.labels[index]   
        return item
    

                #------------------Model---------------

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels = 2
)

                #----------------------device setup----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


                #------------------Data Loaders---------------------

train_dataset = EmailDataset(train_texts, train_labels)
val_dataset = EmailDataset(val_texts, val_labels)
test_dataset = EmailDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE)

                #---------------------Optimizer-------------------------

optimizer = AdamW(model.parameters(), lr = 2e-5)

                #-------------------------Training Loop---------------------- 
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            labels = batch["labels"]
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)   
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

#------------Model evaluation--------------------------

model.eval()
preds = []
true = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        predictions = torch.argmax(outputs.logits, dim=1)
        preds.extend(predictions.cpu().tolist())
        true.extend(batch["labels"].cpu().tolist())

print("\nValidation Results:")
print(classification_report(true, preds))

#------------------Calibration---------------------------

model.eval()

val_logits = []
val_labels = []

with torch.no_grad() :
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        val_logits.append(outputs.logits.cpu())
        val_labels.append(batch["labels"].cpu())

val_logits = torch.cat(val_logits)
val_labels = torch.cat(val_labels)

temperature = torch.nn.Parameter(torch.ones(1))

optimizer_T = torch.optim.LBFGS(
    [temperature],
    lr=0.01,
    max_iter=50
)

def temperature_loss():
    optimizer_T.zero_grad()
    scaled_logits = val_logits / torch.clamp(temperature, min=1e-6)
    loss = F.cross_entropy(scaled_logits, val_labels)
    loss.backward()
    return loss

optimizer_T.step(temperature_loss)

learned_temperature = temperature.item()
print(f"\nLearned temperature: {learned_temperature:.4f}")


with open("email_temperature.json", "w") as f:
    json.dump({"temperature": learned_temperature}, f)


#------------------Test model-------------------------------

test_preds = []
test_true = []

with torch.no_grad() :
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model (
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        predictions = torch.argmax(outputs.logits, dim=1)
        test_preds.extend(predictions.cpu().tolist())
        test_true.extend(batch["labels"].cpu().tolist())

print("\nTest Results:")
print(classification_report(test_true, test_preds))

#-------------------Saving the model---------------------------

model.save_pretrained("email_phishing_model")
tokenizer.save_pretrained("email_phishing_model")