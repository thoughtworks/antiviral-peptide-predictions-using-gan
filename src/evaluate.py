from sklearn import metrics
from sklearn.metrics import confusion_matrix

from src.features.build_features import create_properties_and_plots, create_sequence_properties_dataframe
import pandas as pd


def evaluated(sequences):
    natural_avp = pd.read_csv("")
    seq_properties = create_sequence_properties_dataframe(natural_avp)
    create_properties_and_plots('metadata.csv', '../../reports/')
    pass



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