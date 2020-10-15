import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.files.utils import VALID_AA
# Working directory is: antiviral-peptide-predictions-using-gan
avp = pd.read_csv('data/raw/low_mic_avp.csv', header=0)

vocabulary_avp = []
vocabulary_avp[:0]=VALID_AA
max_len = 50

# ---- Creating the  generator ----
generator_input = keras.Input(shape=(max_len, len(vocabulary_avp)))
x = layers.Dense(128)(generator_input)
x = layers.Dense(128)(x)
x = layers.Dense(128)(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128)(x)
x = layers.LeakyReLU()(x)
x = layers.Dense(128)(x)
x = layers.Dense(20,activation='sigmoid')(x)

generator = keras.models.Model(generator_input, x)
generator.summary()
# ----

# ---- Creating the discriminator ----
dimension = max_len

discriminator_input = layers.Input(shape=(dimension,len(vocabulary_avp)))
x = layers.Dense(64)(discriminator_input)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64)(x)
x = layers.Dense(32)(x)
x = layers.Dense(32)(x)
x = layers.Dropout(0.2)(x)
x = layers.Flatten()(x)
x = layers.Dense(1)(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.01)
discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')

# -----
def to_vocabulary(data):
    '''
    :param data: sequences data containing all the sequences
    :return: this will return a set which contains alphabets which were used for creating sequence
    '''
    all_desc = set() # make a set so that no words gets repeated
    for seq in data:
        for split in seq:
            all_desc.update(split)
    return sorted(all_desc)

vocabulary_avp = to_vocabulary(avp.Sequence)
length_of_vocabulary = len(vocabulary_avp)
print('Vocabulary Size: %d' % length_of_vocabulary)
print("Example of Vocabulary :", (list(vocabulary_avp)))
print("------------------------------------------------------------------------------------------------------------")

# converting words to integers
word_to_index = {} # store all words mapped to integers
index_to_word = {} # store all integers mapped to words

for idx, token in enumerate(vocabulary_avp):
    word_to_index[token] = idx + 1
    index_to_word[idx + 1] = token

print("Word to index example: ",list(word_to_index))
print("Index to word example: ",list(index_to_word))


def convert(data):
    results = np.zeros((len(data), max_len, length_of_vocabulary))
    for sequences in data:
        for i,sequence in enumerate(sequences):
            results[i,word_to_index.get(sequence)] = 1
    return results

converted_data_amp = convert(avp.Sequence)
discriminator_data = []
for i in converted_data_amp:
    discriminator_data.append(i)

percentage_data_for_train = int(len(avp)* (90/100))
discriminator_data = discriminator_data[:percentage_data_for_train]

print("Length of discriminator data: ", len(discriminator_data))

x_train = np.asarray(discriminator_data)
# ----


length = []
for seq in avp.Sequence:
    length.append(len(seq))

max_len = max(length)
print("Maximum Length of Sequence is: ",max_len)

start = 0
max_len = 50



discriminator.trainable = False

gan_input = keras.Input(shape=(max_len,len(vocabulary_avp)))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

iterations = 500
num_to_generate = 300

generated_sequence = []
discriminator_loss = []
adversarial_loss = []

for step in range(iterations):
    random_latent_vectors =  np.random.normal(size = (num_to_generate, max_len, len(vocabulary_avp))) #Samples random points in the latent space

    generated_avps = generator.predict(random_latent_vectors) #Decodes them to fake images

    stop = start + num_to_generate
    real_avps = x_train[start: stop]
    combined_avps = np.concatenate([generated_avps, real_avps])

    labels = np.concatenate([np.zeros((num_to_generate, 1)), np.ones((num_to_generate, 1))])
    labels += 0.05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_avps, labels)

    random_latent_vectors = np.random.normal(size = (num_to_generate, max_len, len(vocabulary_avp)))

    misleading_targets = np.ones((num_to_generate, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += num_to_generate
    if start > len(x_train) - num_to_generate:
        start = 0

    print(step)
    discriminator_loss.append(d_loss)
    adversarial_loss.append(a_loss)
    generated_sequence.append(generated_avps)

print(np.asarray(generated_avps).shape)


sequences = []
string = ""
length = 0
dim = 0
vocab = 0
while length <= (num_to_generate - 1):
    if generated_avps[length,dim,vocab] > 0.8:
        alphabet = index_to_word.get(vocab)
        if alphabet != None:
            #print(length, dim, vocab, alphabet)
            alphabet = str(alphabet)
            string+=alphabet

    vocab += 1
    if vocab == len(vocabulary_avp):
        vocab = 0
        dim += 1

    if dim == max_len:
        dim = 0
        length += 1
        sequences.append(string)
        list1 = []
        string = ""


print(sequences)

os.remove("src/models/simple_gan/generated_simple_gan_low_mic_avp.txt")
for generated_seq in sequences:
    # if len(generated_seq) < max_len:
    next_line = "\n"
    # save sequences
    seq = open("src/models/simple_gan/generated_simple_gan_low_mic_avp.txt", "a+")
    seq.write(generated_seq)
    seq.write(next_line)



def generate_sequences(num=300, max_len = 50):

    random_latent_vectors = np.random.normal(size=(num, max_len, len(vocabulary_avp)))  # Samples random points in the latent space

    generated_avps = generator.predict(random_latent_vectors)
    print(generated_avps.shape)
    return generated_avps







