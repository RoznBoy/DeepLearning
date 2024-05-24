# dataset.py
import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        Args:
            input_file: txt file

        Note:
            1) Load input file and construct character dictionary {index:character}.
            2) Make list of character indices using the dictionary
            3) Split the data into chunks of sequence length 30. 
               You should create targets appropriately.
    """

    def __init__(self, input_file):
        with open(input_file, 'r') as file:
            self.text = file.read()
        
        # Create a character to index dictionary
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}

        # Convert all characters in the text to indices
        self.data = [self.char_to_idx[ch] for ch in self.text]
        self.seq_length = 30  # Fixed sequence length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx: idx + self.seq_length]
        target_seq = self.data[idx + 1: idx + self.seq_length + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)

if __name__ == '__main__':
    dataset = Shakespeare('shakespeare.txt')
    print(len(dataset))
    print(dataset[0])
