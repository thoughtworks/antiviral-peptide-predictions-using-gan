from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = []
amp = []
with open('amp_generated.txt','r') as fin:
    for _ in fin:
        amp.append(_)
        desc = GlobalDescriptor(str(_))
        desc.calculate_all(amide=True)
        data.append(desc.descriptor)
out_arr = np.asarray(data)
m,n,r = out_arr.shape
out_arr = np.column_stack((np.repeat(np.arange(m),n),out_arr.reshape(m*n,-1)))
out_arr = np.delete(out_arr, 0, axis=1)
#print(out_arr)
#print(out_arr.shape)
amp = pd.DataFrame(amp,columns=['Amp'])
amp_descriptors = pd.DataFrame(out_arr,columns=desc.featurenames)
amp_descriptors = pd.concat([amp,amp_descriptors],axis=1)
#print(amp_descriptors.head())
amp_descriptors.to_csv('amp_descriptors.csv')

amp_descriptors.plot(kind='bar',x='Amp',y='pI')
amp_descriptors.plot(kind='bar',x='Amp',y='Charge')

plt.show()