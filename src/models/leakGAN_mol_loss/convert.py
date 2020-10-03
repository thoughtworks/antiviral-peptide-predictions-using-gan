import numpy as np
import pickle
from datetime import date

generated_sample = 'last_generated_sample.npy'

#opening .npy file
file_AMP = np.load('data/corpus_AMP.npy')
file_nonAMP = np.load('data/corpus_nonAMP.npy')
#opening the generated file
genseq = np.load(generated_sample)
#print(genseq)
#getting the encoding character file
vocab_file = "data/chars_AMP.pkl"
w = pickle.load(open(vocab_file,'rb'))
#print(w)
#decoding the seq back
ampb = ''
for i in range(0,50):
    for j in range(0,20):
        index = genseq[i,j]
        #index -= 1
        ampb += str(w[index])
        if j == 19:
            ampb += '\n'
time = str(date.today())
outfile = open("Generated_Seq"+time+".txt", "w")
outfile.write(ampb)
outfile.close()
print(ampb)
#print(len(ampb))






