from sklearn import metrics
from sklearn.metrics import confusion_matrix

from src.features.build_features import create_properties_and_plots, create_sequence_properties_dataframe
import pandas as pd


def evaluate(sequence_file_path):
    sequences = pd.read_csv(sequence_file_path)
    seq_properties = create_sequence_properties_dataframe(sequences)

    # Rules
    temp = seq_properties[seq_properties.net_charge_at_pH7point4 >= -1]
    temp = temp[temp.gravy > -1]
    # temp = temp[temp.helix > 0.35]
    # temp = temp[temp.sheet < 0.3]
    temp = temp[temp.length >= 7]
    temp = temp[temp.length <= 30]
    # temp = temp[temp.molecular_weight <= 2000]
    print(temp)
    return temp




def evaluate_classifier(actual, predicted):
    print("---- Confusion matrix ----")
    print(confusion_matrix(actual, predicted))
    # metrics.accuracy_score(Y_test,predictions)

    print("---- Accuracy score ----")
    print(metrics.accuracy_score(actual, predicted))

    print("---- Balanced accuracy score ----")
    print(metrics.balanced_accuracy_score(actual, predicted))

    print("---- F1 score ----")
    print(metrics.f1_score(actual, predicted))

    print("---- Classification report ----")
    print(metrics.classification_report(actual, predicted))


sequence_file_path = "data/generated/generated_sequences_leakgan.csv"
sequence_file_path = "data/generated/generated_sequences.csv"
sequence_file_path = "data/generated/final_submission.csv"
sequence_file_path = "data/raw/manual_lowMIC.csv"
evaluate(sequence_file_path)
# seq_properties.to_csv("data/sequence_properties/manual_lowMIC/basic_properties.csv")
# seq_properties.to_csv("data/final_submission_basic_properties.csv")

