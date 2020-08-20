from __future__ import print_function, division

import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import pandas as pd

from src.models.utils import *

from keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils

def get_alphabet_list():
    """
    Get all the amino acids from the data files and returns a single flattened out list of amino acids
    
    example:
    Input: ['A', 'C', 'A','G'], ['G', 'A', 'T','D']
    Output: ['A', 'C', 'A','G', 'G', 'A', 'T','D']

    """
    data = pd.read_csv("data/raw/positive_data_unfiltered2807.csv")
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data = additional_filtering_of_sequences(data)
    seq = data['Sequence'].dropna()
    aa_list = list("".join(seq.to_list()))

    return aa_list

def prepare_sequences(alphabet_list, n_alphabets):
    """
    Prepare the sequences used by the Neural Network.
    Takes in the alphabet_list and breaks it into two lists.
    First is a list of subsequences with rolling window of sequence_length and maps it to a numeric value.
    Second is list of alphabet whic follows each subsequence from 1st list.
    This are X and y where X is input sequence and y is next letter given X.
    (This does breaks the structure of a peptide sequence which is provided by its original length.
    Another method that could have been used was to take only those sequence which have same length.
    Still working on how to accomodate varying length to pass original sequence as it is.)
    
    example:
    Input:
    -----
    alphabet_list = ['A', 'C', 'A','G', 'G', 'A', 'T','D']
    sequence_length = 4
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3,'D': 4}
    
    Output:
    ------
    mapped alphabet list = [0, 1, 0, 2, 2, 0, 3, 4]
    X = [[0, 1, 0, 2], [1, 0, 2, 2], [0, 2, 2, 0], [2, 2, 0, 3], [2, 0, 3, 4]
    y = [2, 0, 3, 4]

    this is noramalised to be b/w -1 and 1

    """
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

def generate_peptide(model, alphabet_list, latent_dim, n_alphabets):
    """
    Generate peptide from the neural network based on a sequence of amino acids.
    Based on random noise generate a sequence of defined sequence length.
    Convert this numeric sequence to sequence of amino acids using inverse mapping.
    """
    
    # Get alphabet names to store in a dictionary
    alphabets = sorted(set(alphabet_list))

    #create inverse mapping from int to alphabet
    int_to_note = dict(zip(range(len(alphabets)), alphabets))
        
    # Use random noise to generate sequences
    noise = np.random.normal(0, 1, (1, latent_dim))
    predictions = model.predict(noise)
        
    pred_seq = [(x*n_alphabets/2)+(n_alphabets/2) for x in predictions[0]]
    pred_seq = [int_to_note[int(x)] for x in pred_seq]

    return pred_seq


class GAN():
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 1000
        self.disc_loss = []
        self.gen_loss =[]
        
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates peptide sequences
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(generated_seq)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):

        model = Sequential()
        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        seq = Input(shape=self.seq_shape)
        validity = model(seq)

        return Model(seq, validity)
      
    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))
        model.summary()
        
        noise = Input(shape=(self.latent_dim,))
        seq = model(noise)

        return Model(noise, seq)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load and convert the data
        alphabet_list = get_alphabet_list()
        n_alphabets = len(set(alphabet_list))
        X_train, _ = prepare_sequences(alphabet_list, n_alphabets)

        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Training the model
        for epoch in range(epochs):

            # Training the discriminator
            # Select a random batch of note sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]

            #noise = np.random.choice(range(484), (batch_size, self.latent_dim))
            #noise = (noise-242)/242
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new peptide sequences
            gen_seqs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #  Training the Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)

            # Print the progress and save into loss lists
            if epoch % sample_interval == 0:
              print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
              self.disc_loss.append(d_loss[0])
              self.gen_loss.append(g_loss)
        
        self.generator.save('GANmodel.h5')
        pred_seq = self.generate(alphabet_list)
        self.plot_loss()
        
    def generate(self, alphabet_list):
        """ Same as generate_peptide function """
        
        # Get alphabet names to store in a dictionary
        alphabets = sorted(set(alphabet_list))
        n_alphabets = len(set(alphabet_list))
        int_to_note = dict(zip(range(len(alphabets)), alphabets))
        
        # Use random noise to generate sequences
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)
        
        pred_seq = [(x*n_alphabets/2)+(n_alphabets/2) for x in predictions[0]]
        pred_seq = [int_to_note[int(x)] for x in pred_seq]

        return pred_seq
        
    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.savefig('src/visualization/GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()

if __name__ == '__main__':
  gan = GAN(rows=100)    
  gan.train(epochs=5000, batch_size=32, sample_interval=1)


"""
Get following error only when running these scripts on system with GPU.

Till early versions of keras, keras had seperate layer with CUDA implementation namely, CuDNNRNN, CuDNNLSTM and CuDNNGRU.
Now there are only RNN, LSTM and GRU layers and if GPU is being used then automatically CUDA implementations of these layers are used.
This error is because CuDNNRNN is not found as I was trying to run them on GPU.

UnknownError:    Fail to find the dnn implementation.
	 [[{{node CudnnRNN}}]]
	 [[sequential/lstm/PartitionedCall]] [Op:__inference_train_function_13130]

Function call stack:
train_function -> train_function -> train_function
"""
