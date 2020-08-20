import glob
import numpy as np
import pandas as pd

from utils import *

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, History

def train_network():
    """ Train a Neural Network to generate music """
    # Get notes from midi files
    alphabet_list = get_alphabet_list()

    # Get the number of pitch names
    n_alphabets = len(set(alphabet_list))

    # Convert notes into numerical input
    network_input, network_output = prepare_sequences(alphabet_list, n_alphabets)

    # Set up the model
    model = create_network(network_input, n_alphabets)
    history = History()
    
    # Fit the model
    n_epochs = 2500
    model.summary()
    model.fit(network_input, network_output, callbacks=[history], epochs=n_epochs, batch_size=64)
    model.save('LSTMmodel.h5')
    
    # Use the model to generate a midi
    prediction_output = generate_peptide(model, alphabet_list, network_input, n_alphabets)
    
    # Plot the model losses
    pd.DataFrame(history.history).plot()
    plt.savefig('LSTM_Loss_per_Epoch.png', transparent=True)
    plt.close()
    
def get_alphabet_list():
    """ Get all the amino acids from the data files """
    data = pd.read_csv("data/raw/positive_data_unfiltered2807.csv")
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data = additional_filtering_of_sequences(data)
    seq = data['Sequence'].dropna()
    aa_list = list("".join(seq.to_list()))

    return aa_list
  
def prepare_sequences(alphabet_list, n_alphabets):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all alphabet names
    alphabets = sorted(set(alphabet_list))

     # create a dictionary to map alphabet to integers
    alphabet_to_int = dict(zip(alphabets, range(len(alphabets))))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(alphabet_list) - sequence_length, 1):
        sequence_in = alphabet_list[i:i + sequence_length]
        sequence_out = alphabet_list[i + sequence_length]
        network_input.append([alphabet_to_int[char] for char in sequence_in])
        network_output.append(alphabet_to_int[sequence_out])

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input between 0 and 1
    network_input = network_input / float(n_alphabets)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)
  
def create_network(network_input, n_alphabets):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(512,input_shape=(network_input.shape[1], network_input.shape[2]),return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_alphabets))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
    
def generate_peptide(model, alphabet_list, network_input, n_alphabets):
    """ Generate peptide from the neural network based on a sequence of amino acids """
    
    # pick a random sequence from the input as a starting point for the prediction
    alphabets = sorted(set(alphabet_list))
    
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict(zip(range(len(alphabets)), alphabets))

    pattern = network_input[start]
    prediction_output = []

    # generate sequence of length 500
    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern = np.append(pattern,(index  / float(n_alphabets)))
        pattern = pattern[1:len(pattern)]

    return prediction_output
    
if __name__ == '__main__':
    train_network()
