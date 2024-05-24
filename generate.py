import torch
import numpy as np
from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate(model, seed_characters, temperature, dataset, device, length=100):
    model.eval()
    chars = dataset.chars
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char

    input = torch.tensor([char_to_idx[c] for c in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):  # LSTM case
        hidden = tuple([h.to(device) for h in hidden])
    else:  # RNN case
        hidden = hidden.to(device)

    generated_text = seed_characters

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input, hidden)
            output = output / temperature
            probabilities = torch.softmax(output[-1], dim=-1).cpu().numpy()
            char_idx = np.random.choice(len(chars), p=probabilities)
            char = idx_to_char[char_idx]
            generated_text += char
            input = torch.tensor([[char_idx]], dtype=torch.long).to(device)

    return generated_text

if __name__ == '__main__':
    # Load the dataset and model
    dataset = Shakespeare('shakespeare_train.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(dataset.chars)
    hidden_size = 128
    n_layers = 2

    model = CharLSTM(vocab_size, hidden_size, n_layers).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))

    # Generate text
    seed_characters = 'ROMEO: '
    temperature = 1.0
    print(generate(model, seed_characters, temperature, dataset, device, length=100))

    # Generate with different temperatures
    for temp in [0.5, 1.0, 1.5]:
        print(f'\nTemperature: {temp}')
        print(generate(model, seed_characters, temp, dataset, device, length=100))
