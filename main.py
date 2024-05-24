import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):  # LSTM case
            hidden = tuple([h.to(device) for h in hidden])
        else:  # RNN case
            hidden = hidden.to(device)
        
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):  # LSTM case
                hidden = tuple([h.to(device) for h in hidden])
            else:  # RNN case
                hidden = hidden.to(device)

            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main():
    # Load dataset
    dataset = Shakespeare('shakespeare_train.txt')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate models
    vocab_size = len(dataset.chars)
    hidden_size = 128
    n_layers = 2

    rnn_model = CharRNN(vocab_size, hidden_size, n_layers).to(device)
    lstm_model = CharLSTM(vocab_size, hidden_size, n_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.002)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.002)

    # Training
    n_epochs = 20
    rnn_train_losses, rnn_val_losses = [], []
    lstm_train_losses, lstm_val_losses = [], []

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(n_epochs):
        rnn_train_loss = train(rnn_model, train_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        lstm_train_loss = train(lstm_model, train_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)
        
        rnn_train_losses.append(rnn_train_loss)
        rnn_val_losses.append(rnn_val_loss)
        lstm_train_losses.append(lstm_train_loss)
        lstm_val_losses.append(lstm_val_loss)

        print(f'Epoch {epoch+1}/{n_epochs}, RNN Train Loss: {rnn_train_loss:.4f}, RNN Val Loss: {rnn_val_loss:.4f}, LSTM Train Loss: {lstm_train_loss:.4f}, LSTM Val Loss: {lstm_val_loss:.4f}')

        # Save the best model
        if lstm_val_loss < best_val_loss:
            best_val_loss = lstm_val_loss
            best_model = lstm_model.state_dict()

    if best_model:
        torch.save(best_model, 'best_model.pth')

    # Plot the losses
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rnn_train_losses, label='RNN Train Loss')
    plt.plot(rnn_val_losses, label='RNN Val Loss')
    plt.legend()
    plt.title('RNN Losses')
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_train_losses, label='LSTM Train Loss')
    plt.plot(lstm_val_losses, label='LSTM Val Loss')
    plt.legend()
    plt.title('LSTM Losses')
    
    plt.show()

if __name__ == '__main__':
    main()
