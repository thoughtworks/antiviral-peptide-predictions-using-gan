import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree, metrics
import matplotlib.pyplot as plt

from src.features.build_features import create_sequence_properties_dataframe

os.getcwd()
os.chdir('/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan')

avp = pd.read_csv("data/raw/manual_lowMIC.csv")
non_avp = pd.read_csv("data/raw/AMP_nonAVP_filtered_negative.csv")

avp_seq_properties = create_sequence_properties_dataframe(avp)
non_avp_seq_properties = create_sequence_properties_dataframe(non_avp)

avp_seq_properties['Activity'] = 1
non_avp_seq_properties['Activity'] = 0

data = avp_seq_properties
data = data.append(non_avp_seq_properties)
params = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'gravy', 'net_charge_at_pH7point4']
# Creating a DT classifier

X_train, X_test, Y_train, Y_test = train_test_split(data[params], data['Activity'], random_state=0)

clf = DecisionTreeClassifier(max_depth = 5, random_state = 0, criterion="entropy")
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
print(confusion_matrix(Y_test,predictions))
# metrics.accuracy_score(Y_test,predictions)
metrics.balanced_accuracy_score(Y_test,predictions)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,2), dpi=3000)
tree.plot_tree(clf, feature_names=params, class_names=['0','1'],filled=True)
fig.savefig('DT_depth5.png')

"""
highly active - manually curated - data/raw/manual_lowMIC.csv
avp - low_mic_90perc - AVPpred_Low_MIC_data_filtered_90perc.csv
nonAVP_AMP - 90perc - raw/AMP_nonAVP_filtered_negative.csv
"""
