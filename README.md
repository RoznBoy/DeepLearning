# Character-Level Language Modeling

This project implements character-level language modeling using Vanilla RNN and LSTM on the Shakespeare dataset.

## Files
- `dataset.py`: Contains the dataset class for processing the Shakespeare dataset.
- `model.py`: Defines the CharRNN and CharLSTM models.
- `main.py`: Handles the training and validation of the models.
- `generate.py`: Generates text using a trained model.

## Results
### Loss Plots
![Figure_1](https://github.com/RoznBoy/DeepLearning/assets/154126402/b2460470-654a-4f5e-9490-a038e1ddff07)


### Analysis
The results show that the LSTM model generally performs better than the Vanilla RNN model in terms of validation loss. The average loss values for both training and validation datasets are plotted in the figure above.

- **RNN**: The training and validation loss for the Vanilla RNN model tends to be higher and more unstable compared to the LSTM model.
- **LSTM**: The LSTM model shows more stable and lower training and validation loss values, indicating better performance in learning and generalization.

## Text Generation
The trained models are used to generate text based on seed characters. The following text samples are generated using the LSTM model with different temperature settings.

### Temperature Analysis

The softmax function with a temperature parameter \(T\) can be written as:

$$
y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}}
$$

We experimented with different temperatures to see how it affects the generated text.

### Temperature 0.5
Lower temperature results in more conservative and repetitive text generation as it makes the probability distribution sharper.


### Temperature 1.0
Moderate temperature results in balanced and coherent text generation.


### Temperature 1.5
Higher temperature results in more diverse and creative text generation, but it might also lead to less coherent text.


### Analysis
- **Low Temperature (0.5)**: The model becomes conservative, generating repetitive and less diverse text. The characters with the highest probabilities dominate, making the text more predictable.
- **Moderate Temperature (1.0)**: The model produces a good balance between diversity and coherence. This temperature generally generates the most plausible and readable text.
- **High Temperature (1.5)**: The model becomes more creative and diverse, but the text can become less coherent and more random. The probability distribution flattens, allowing for more exploration of less likely characters.

Adjusting the temperature parameter helps in controlling the creativity and coherence of the generated text. Lower temperatures make the text safe and repetitive, while higher temperatures introduce more variability at the cost of coherence.

## Dependencies
- Python 3.x
- PyTorch
- Matplotlib
- NumPy

## How to Run
1. Download the Shakespeare dataset (`shakespeare_train.txt`) and place it in the same directory as the scripts.
2. Execute `main.py` to train the models.
3. Execute `generate.py` to generate text with the trained model.

