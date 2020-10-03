import pandas as pd
import numpy as np
import pickle

positive_data = "../../../data/raw/AVP_data.csv"
negative_data = "../../../data/raw/non_AVP_data.csv"
vocab_file = "./data/chars_AMP.pkl"
seq_len = 20

avp = pd.read_csv(positive_data)
non_avp = pd.read_csv(negative_data)

avp_fil = avp[avp.Sequence.apply(lambda s:len(s)==seq_len)] 
non_avp_fil = non_avp[non_avp.Sequence.apply(lambda s:len(s)==seq_len)]

with open(vocab_file, 'rb') as f:
	ch = pickle.load(f)

avp_int = np.array([[ch.index(a) for a in s] for s in avp_fil.Sequence], dtype="int32")
non_avp_int = np.array([[ch.index(a) for a in s] for s in non_avp_fil.Sequence], dtype="int32")

avp_fil.to_csv("./data/AVP_data.txt", index=False, header=False)
non_avp_fil.to_csv("./data/non_AVP_data.txt", index=False, header=False)

np.save("./data/corpus_AVP.npy", avp_int)
np.save("./data/corpus_nonAVP.npy", non_avp_fil)
