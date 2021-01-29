import pandas as pd

def add_spaces_after_each_aa(file_path="data/"):
    df = pd.read_csv(file_path)
    result = pd.DataFrame(columns=['Sequence'])
    for seq in df.Sequence:
        t = (" ".join(seq))
        result = result.append(pd.DataFrame([t], columns=['Sequence']), ignore_index=True)
    result['Sequence'].to_csv("data/input_sequences.txt", index=False)
    return result


def remove_spaces_from_the_generated_peptides(gan_type, file_path="save/test_file.txt", save=True):
    df = pd.read_csv(file_path, header=None)
    result = pd.DataFrame(columns=['Sequence'])
    for seq in df[0]:
        t = seq.replace(" ", "").upper()
        result = result.append(pd.DataFrame([t], columns=['Sequence']))
    if save:
        result['Sequence'].to_csv("data/generated_sequences_"+ gan_type +".txt", index=False)
    return result

import pickle

pickle.dump(generator,"leakGAN_model")