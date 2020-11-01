import pandas as pd
import os

from sklearn.model_selection import train_test_split
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

highly_active_avp = pd.read_csv("data/raw/manual_lowMIC.csv") #130
medium_avp = pd.read_csv("data/filtered/AVPpred_Low_MIC_data_filtered_90perc.csv") #424
non_avp = pd.read_csv("data/raw/AMP_nonAVP_filtered_negative.csv") #2368
avp_pred_neg_data = pd.read_csv("data/raw/non_AVP_data.csv") #444

highly_activeavp_seq_properties = create_sequence_properties_dataframe(highly_active_avp)
medium_avp_seq_properties = create_sequence_properties_dataframe(medium_avp)
non_avp_seq_properties = create_sequence_properties_dataframe(non_avp)
non_avp_pred_seq_properties = create_sequence_properties_dataframe(avp_pred_neg_data)


highly_activeavp_seq_properties['Activity'] = 1
medium_avp_seq_properties['Activity'] = 1
non_avp_seq_properties['Activity'] = 0
non_avp_pred_seq_properties['Activity'] = 0

highly_activeavp_seq_properties = pd.concat([highly_activeavp_seq_properties.drop(['aa_percentages'], axis=1), highly_activeavp_seq_properties['aa_percentages'].apply(pd.Series)], axis=1)
medium_avp_seq_properties = pd.concat([medium_avp_seq_properties.drop(['aa_percentages'], axis=1), medium_avp_seq_properties['aa_percentages'].apply(pd.Series)], axis=1)
non_avp_seq_properties = pd.concat([non_avp_seq_properties.drop(['aa_percentages'], axis=1), non_avp_seq_properties['aa_percentages'].apply(pd.Series)], axis=1)
non_avp_pred_seq_properties = pd.concat([non_avp_pred_seq_properties.drop(['aa_percentages'], axis=1), non_avp_pred_seq_properties['aa_percentages'].apply(pd.Series)], axis=1)

basic_params = highly_activeavp_seq_properties.columns.to_list()

# ----------------------------------- Loading other properties ---------------------------------------------------------
# Load composition based properties
to_drop=['id','Sequence']
highly_activeavp_comp = pd.read_csv("data/sequence_properties/manual_lowMIC/composition_based.csv").drop(to_drop, axis=1)
medium_avp_comp = pd.read_csv("data/sequence_properties/AVPpred_Low_MIC_data_filtered_90perc/composition_based.csv").drop(to_drop, axis=1)
non_avp_comp = pd.read_csv("data/sequence_properties/AMP_nonAVP_filtered_negative/composition_based.csv").iloc[:,:-3].drop(to_drop, axis=1)
composition_params = highly_activeavp_comp.columns.to_list()

# Load dipeptide properies
to_drop=['ID']
highly_activeavp_dipeptide = pd.read_csv("data/sequence_properties/manual_lowMIC/final_dipeptide_result.csv").drop(to_drop, axis=1)
medium_avp_dipeptide = pd.read_csv("data/sequence_properties/AVPpred_Low_MIC_data_filtered_90perc/final_dipeptide_result.csv").drop(to_drop, axis=1)
non_avp_dipeptide = pd.read_csv("data/sequence_properties/AMP_nonAVP_filtered_negative/final_dipeptide_result.csv").drop(to_drop, axis=1)
non_avp_pred_seq_dipeptide = pd.read_csv("data/sequence_properties/AVPpred_nonAVP/final_dipeptide_result.csv").drop(to_drop, axis=1)
dipeptide_params = highly_activeavp_dipeptide.columns.to_list()

"""

# Load tripeptide properties
to_drop=['ID']
highly_activeavp_tripeptide = pd.read_csv("data/sequence_properties/manual_lowMIC/final_tripep_composition_result.csv").drop(to_drop, axis=1)
medium_avp_tripeptide = pd.read_csv("data/sequence_properties/AVPpred_Low_MIC_data_filtered_90perc/final_tripep_composition_result.csv").drop(to_drop, axis=1)
non_avp_tripeptide = pd.read_csv("data/sequence_properties/AMP_nonAVP_filtered_negative/final_tripep_composition_result.csv").drop(to_drop, axis=1)
non_avp_tripeptide = pd.read_csv("data/sequence_properties/AVPpred_nonAVP/final_tripep_result.csv").drop(to_drop, axis=1)

tripeptide_params = highly_active_avp_all_prop.columns.to_list()
"""

"""

# Load autocorrelation properties
to_drop=['Sequence']
highly_activeavp_autocorrelation = pd.read_csv("data/sequence_properties/manual_lowMIC/final_autocorr_result.csv").drop(to_drop, axis=1).reset_index()
medium_avp_autocorrelation = pd.read_csv("data/sequence_properties/AVPpred_Low_MIC_data_filtered_90perc/final_autocorr_result.csv").drop(to_drop, axis=1).reset_index()
non_avp_autocorrelation = pd.read_csv("data/sequence_properties/AMP_nonAVP_filtered_negative/final_autocorr_result.csv").drop(to_drop, axis=1).reset_index()

autocorrelation_params = highly_activeavp_autocorrelation.columns.to_list()
"""

# Load shanon entropy
to_drop = ['ID', 'Sequence']
highly_activeavp_se = pd.read_csv("data/sequence_properties/manual_lowMIC/final_SE_residue_result.csv").drop(['Sequence'], axis=1)
medium_avp_se = pd.read_csv("data/sequence_properties/AVPpred_Low_MIC_data_filtered_90perc/final_SE_residue_result.csv").drop(to_drop, axis=1)
non_avp_se = pd.read_csv("data/sequence_properties/AMP_nonAVP_filtered_negative/final_SE_residue_result.csv").drop(to_drop, axis=1)
non_avp_pred_se = pd.read_csv("data/sequence_properties/AVPpred_nonAVP/final_SE_residue_result.csv").drop(to_drop, axis=1)

shanon_entropy_params = highly_activeavp_se.columns.to_list()


# ------------------------------------------ Combine all properties ---------------------------------------------
highly_active_all_props_combined = [highly_activeavp_seq_properties.set_index(highly_activeavp_comp.index), highly_activeavp_comp, highly_activeavp_dipeptide, highly_activeavp_se]
medium_avp_all_props_combined = [medium_avp_seq_properties.set_index(medium_avp_comp.index), medium_avp_comp, medium_avp_dipeptide, medium_avp_se]
non_avp_all_props_combined = [non_avp_seq_properties.set_index(non_avp_dipeptide.index), non_avp_comp, non_avp_dipeptide, non_avp_se]
non_avp_pred_all_props_combined = [non_avp_pred_seq_properties.set_index(non_avp_pred_seq_dipeptide.index), non_avp_pred_seq_dipeptide, non_avp_pred_se]


highly_active_avp_all_prop = pd.concat(highly_active_all_props_combined, axis=1)#.drop('index', axis=1)
medium_avp_all_prop = pd.concat(medium_avp_all_props_combined, axis=1)#.drop('index', axis=1)
non_avp_all_prop = pd.concat(non_avp_all_props_combined, axis=1)#.drop('index', axis=1)
non_avp_pred_all_prop = pd.concat(non_avp_pred_all_props_combined, axis=1)#.drop('index', axis=1)

highly_active_avp_all_prop.columns
medium_avp_all_prop.columns
non_avp_all_prop.columns
non_avp_pred_all_prop.columns

# ------------------------------------- Creating the final data --------------------------------------------------------
data = highly_active_avp_all_prop
data = data.append(medium_avp_all_prop)
data = data.append(non_avp_all_prop)
# ------------------------------------------ Selecting relevant parameters ---------------------------------------------

"""
All parameters:

params = shanon_entropy_params + autocorrelation_params + tripeptide_params + dipeptide_params + composition_params + basic_params

params = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'gravy', 'net_charge_at_pH7point4','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
params = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
params = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'gravy',
       'net_charge_at_pH7point4', 'M wt (Da)', 'Number of Amnio Acids', 'Charged (DEKHR)',
       'Aliphatic (ILV)', 'Aromatic (FHWY)', 'Polar (DERKQN)',
       'Neutral (AGHPSTY)', 'Hydrophobic (CFILMVW)', '+ charged (KRH)',
       '- charged (DE)', 'Tiny (ACDGST)', 'Small (EHILKMNPQV)',
       'Large (FRWY)','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
"""
basic_params = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'gravy', 'net_charge_at_pH7point4','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

params = basic_params + composition_params + dipeptide_params + shanon_entropy_params
params = basic_params
params = composition_params
params = dipeptide_params
params = shanon_entropy_params

# -------------------------------------------- Create classification models --------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(data[params], data['Activity'], random_state=0)

# ---- Standarise the data ----
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ---- Decision Tree classifier ----
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(max_depth = 10, random_state = 0, criterion="entropy")
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
evaluate_classifier(Y_test, predictions)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,2), dpi=3000)
tree.plot_tree(clf, feature_names=params, class_names=['0','1'],filled=True)
fig.savefig('DT_depth10_all_params.png')

# Showing feature importance
importance = clf.feature_importances_
# summarize feature importance
imp_features = pd.DataFrame(columns = ['feature', 'score'])
for i,v in enumerate(importance):
	# print('Feature: %0d, Score: %.5f' % (i,v))
	imp_features = imp_features.append(pd.DataFrame([[params[i],v]], columns = ['feature', 'score']))
# plot feature importance

imp_features.sort_values(by='score', ascending=False)
plt.bar([x for x in range(len(importance))], importance)
plt.show()

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
#
# # xgboost for feature importance on a classification problem
# from sklearn.datasets import make_classification
# from xgboost import XGBClassifier
# from matplotlib import pyplot
# # define dataset
# # X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# # define the model
# model = XGBClassifier()
# # fit the model
# model.fit(X_train, Y_train)
# predictions = gnb_classifier.predict(X_test)
# evaluate_classifier(Y_test, predictions)
# # get importance
# importance = model.feature_importances_
# # summarize feature importance
# imp_features = pd.DataFrame(columns = ['feature', 'score'])
# for i,v in enumerate(importance):
# 	# print('Feature: %0d, Score: %.5f' % (i,v))
# 	imp_features = imp_features.append(pd.DataFrame([[params[i],v]], columns = ['feature', 'score']))
#
# # plot feature importance
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()

# Classify the generated sequences:
generated_sequences = pd.read_csv("data/sequence_properties/submission_sequences_3010/generated_sequences.csv")

generated_sequences_properties = create_sequence_properties_dataframe(generated_sequences)
generated_sequences_properties = pd.concat([generated_sequences_properties.drop(['aa_percentages'], axis=1), generated_sequences_properties['aa_percentages'].apply(pd.Series)], axis=1)
to_drop=['Sequences']
generated_sequences_comp = pd.read_csv("data/sequence_properties/submission_sequences_3010/composition_based.csv").drop(to_drop, axis=1)
to_drop=['ID']
generated_sequences_dipeptide = pd.read_csv("data/sequence_properties/submission_sequences_3010/final_dipeptide_result.csv").drop(to_drop, axis=1)
to_drop = ['ID', 'Sequence']
generated_sequences_se = pd.read_csv("data/sequence_properties/submission_sequences_3010/final_SE_residue_result.csv").drop(['Sequence'], axis=1)
generated_sequences_all_props_combined = [generated_sequences_properties.set_index(generated_sequences_comp.index), generated_sequences_comp, generated_sequences_dipeptide, generated_sequences_se]
generated_sequences_all_prop = pd.concat(generated_sequences_all_props_combined, axis=1)

to_predict = generated_sequences_all_prop[params]

generated_predictions = gnb_classifier.predict(to_predict)
generated_predictions = clf.predict(to_predict)
generated_predictions = mlp_clf.predict(to_predict)
generated_predictions = rf_classifier.predict(to_predict)

