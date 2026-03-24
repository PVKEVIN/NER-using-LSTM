# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

The objective of this project is to develop a Bidirectional Long Short-Term Memory (BiLSTM) model for Name Recognition in sentences. The model will perform Named Entity Recognition (NER) by identifying and classifying person names within input text sequences. Given a sentence, the model should label each word using sequence tagging (e.g., B-PER, I-PER, O) to determine whether it is part of a person’s name or not. The performance of the model will be evaluated using metrics such as accuracy and validation loss.

## DESIGN STEPS

### Step 1:
Import the necessary libraries.

### Step 2:
Load the dataset and use DataLoader to batch the dataset

### Step 3:
Create a class to define the Long Short Term Memory Neural Network, in the class define the forward function

### Step 4:
Initialize the model and get a model summary

### STEP 5:
Initialize the loss function MSELoss and Optimizier

### STEP 6:
Create a function to train the model and call it to train the model.

### STEP 7:
Test the model using the test_loader.

### Step 8:
Display the results.


## PROGRAM
### Name: Kevin P
### Register Number: 212224040159

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_dim)
        self.drop = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2,tagset_size)

    def forward(self, input_ids):
        x = self.embed(x)
        x = self.drop(x)
        x, _ = self.lstm(x)
        x = self.fc(x)

        return x
```

```python
model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

```

```python
# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
   
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
                val_loss += loss.item()

            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, val loss = {val_loss:.4f}") 


    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="941" height="647" alt="image" src="https://github.com/user-attachments/assets/9078726e-b205-4694-84c3-38e3f6c74777" />


### Sample Text Prediction

<img width="345" height="555" alt="image" src="https://github.com/user-attachments/assets/f2a85b40-20dc-46ab-85ce-a04bb32942de" />

<img width="297" height="578" alt="image" src="https://github.com/user-attachments/assets/10d6064e-d82b-4d1f-b89e-85e91b234a19" />


<img width="490" height="433" alt="image" src="https://github.com/user-attachments/assets/f98d292b-a154-4b9e-b651-e1df63e8cf3a" />


## RESULT
Thus, A Long Short Term Memory Neural Network model is implemented successfully for recognizing the named entities in the text.
