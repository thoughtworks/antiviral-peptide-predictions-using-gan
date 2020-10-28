import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

from src.evaluate import evaluate_classifier
from src.features.build_features import create_sequence_properties_dataframe

os.getcwd()
os.chdir('/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan')

"""
Datasets:
highly active - manually curated : data/raw/manual_lowMIC.csv
avp - low_mic_90perc : data/filtered/AVPpred_Low_MIC_data_filtered_90perc.csv
nonAVP_AMP - 90perc : data/raw/AMP_nonAVP_filtered_negative.csv
"""

highly_active_avp = pd.read_csv("data/raw/manual_lowMIC.csv")
medium_avp = pd.read_csv("data/filtered/AVPpred_Low_MIC_data_filtered_90perc.csv")
non_avp = pd.read_csv("data/raw/AMP_nonAVP_filtered_negative.csv")

highly_activeavp_seq_properties = create_sequence_properties_dataframe(highly_active_avp)
non_avp_seq_properties = create_sequence_properties_dataframe(non_avp)

# Upload composition based properties


highly_activeavp_seq_properties['Activity'] = 1
non_avp_seq_properties['Activity'] = 0

highly_activeavp_seq_properties = pd.concat([highly_activeavp_seq_properties.drop(['aa_percentages'], axis=1), highly_activeavp_seq_properties['aa_percentages'].apply(pd.Series)], axis=1)
non_avp_seq_properties = pd.concat([non_avp_seq_properties.drop(['aa_percentages'], axis=1), non_avp_seq_properties['aa_percentages'].apply(pd.Series)], axis=1)

data = highly_activeavp_seq_properties
data = data.append(non_avp_seq_properties)
params = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'gravy', 'net_charge_at_pH7point4','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# Creating a DT classifier

X_train, X_test, Y_train, Y_test = train_test_split(data[params], data['Activity'], random_state=0)

# ---- Standarise the data ----
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ---- Decision Tree classifier ----
clf = DecisionTreeClassifier(max_depth = 50, random_state = 0, criterion="entropy")
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
evaluate_classifier(Y_test, predictions)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,2), dpi=3000)
tree.plot_tree(clf, feature_names=params, class_names=['0','1'],filled=True)
fig.savefig('DT_depth50_withAA.png')

# ---- plot DT ----
# import graphviz
# # DOT data
# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 feature_names=params,
#                                 class_names=['0','1'],
#                                 filled=True)
#
# # Draw graph
# graph = graphviz.Source(dot_data, format="png")
# graph
#
# graph.render("decision_tree_graphivz")
#
# from dtreeviz.trees import dtreeviz



print("----- Logistic Regression ------")
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, Y_train)
predictions = lr_classifier.predict(X_test)
evaluate_classifier(Y_test, predictions)

print("----- RFC ------")
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(30)
rf_classifier.fit(X_train, Y_train)
predictions = rf_classifier.predict(X_test)
evaluate_classifier(Y_test, predictions)

# probability of the data point belonging to the class
predictions = rf_classifier.predict_proba(X_test)

print("----- DL ------")
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 20), random_state=1)
mlp_clf.fit(X_train, Y_train)
predictions = mlp_clf.predict(X_test)
evaluate_classifier(Y_test, predictions)

print("----- SVM ------")
from sklearn.svm import SVC
svc_classifier = SVC(kernel='poly')
svc_classifier.fit(X_train, Y_train)
predictions = svc_classifier.predict(X_test)
evaluate_classifier(Y_test, predictions)

print("----- KNN ------")
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, Y_train)
predictions = knn_classifier.predict(X_test)
evaluate_classifier(Y_test, predictions)

print("---- Gaussian Naive Bayes ----")
from sklearn.naive_bayes import GaussianNB
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, Y_train)
predictions = gnb_classifier.predict(X_test)
evaluate_classifier(Y_test, predictions)

