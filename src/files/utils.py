import pandas as pd
import random

def get_positive_amp_data(path):
    return pd.read_csv(path, sep=",", header=0).reset_index().drop('index', axis=1)[['Sequence', 'Activity']]

OTHER_ALPHABETS = "UOXBZJ"

def contains(other_alphabets, seq):
    for o in str(other_alphabets):
        if o in str(seq):
            return True
    return False

def trim_all(strings):
    return list(set(value.strip().strip(',').lower() for value in strings))

def sequence_filtering(data):
    sequences = data[data.apply(lambda r: not contains(OTHER_ALPHABETS, r['Sequence']), axis=1)]
    sequences = sequences[sequences.apply(lambda r: not str(r['Sequence']) == 'nan', axis=1)]
    sequences['Sequence'] = sequences['Sequence'].apply(lambda x: x.upper())
    return sequences

def massage_camp_data(data):
    sequences = sequence_filtering(data)
    sequences['Activity'] = sequences['Activity'].apply(lambda x: trim_all(str(x).split(',')))
    # sequences['Activity'] = sequences['Activity'].apply(lambda x: x.remove(''))
    sequences = sequences.reset_index().drop('index', axis=1)
    for i in range(sequences.__len__()):
        while '' in sequences.loc[i, 'Activity']:
            sequences.loc[i, 'Activity'].remove('')
    return sequences

def generate_random_sequences(number_of_sequences=100, max_length=100):
    valid_aa = 'ACDEFGHIKLMNPQRSTVWY'
    r = pd.DataFrame(columns=['Sequence'])
    for i in range(number_of_sequences):
        seq_length = random.randrange(0,max_length)
        seq_list = random.choices(valid_aa, k=seq_length)
        seq = pd.DataFrame([''.join(seq_list)], columns = ['Sequence'])
        r = r.append(seq, ignore_index=True)
    return r

a = generate_random_sequences()
a.to_csv('data/generated/generated_random_seq.csv', index=False)