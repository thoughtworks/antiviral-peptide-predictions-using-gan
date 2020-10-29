import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import os
from datetime import datetime


def create_sequence_properties_dataframe(sequences):
    print("---- Creating properties for the all data. This may take a few mins depending on data size ----")
    params = ['sequence', 'aa_counts', 'aa_percentages', 'molecular_weight', 'aromaticity', 'instability_index',
              'isoelectric_point', 'sec_struc', 'helix', 'turn', 'sheet', 'epsilon_prot', 'with_reduced_cysteines',
              'with_disulfid_bridges', 'gravy', 'flexibility','net_charge_at_pH7point4']

    seq_properties = pd.DataFrame(columns=params)

    for seq in sequences.Sequence:
        X = ProteinAnalysis(seq)
        aa_counts = X.count_amino_acids()
        aa_percentages = X.get_amino_acids_percent()
        molecular_weight = X.molecular_weight()
        aromaticity = X.aromaticity()
        instability_index = X.instability_index()
        isoelectric_point = X.isoelectric_point()
        sec_struc = X.secondary_structure_fraction()
        helix = sec_struc[0]
        turn = sec_struc[1]
        sheet = sec_struc[2]
        epsilon_prot = X.molar_extinction_coefficient()
        with_reduced_cysteines = epsilon_prot[0]
        with_disulfid_bridges = epsilon_prot[1]
        gravy = X.gravy() # hydrophobicity related
        flexibility = X.flexibility()
        # X.protein_scale()
        net_charge_at_pH7point4 = X.charge_at_pH(7.4)

        row = pd.DataFrame([[seq, aa_counts, aa_percentages, molecular_weight, aromaticity, instability_index,
                             isoelectric_point, sec_struc, helix, turn, sheet, epsilon_prot, with_reduced_cysteines,
                             with_disulfid_bridges, gravy, flexibility, net_charge_at_pH7point4]], columns=params)
        seq_properties = seq_properties.append(row)
    return seq_properties

# ---- Visualizations on seq properties ----

def create_distributions(data, properties, save=False, saving_dir="../reports/figures/distribution_" ):
    kwargs = dict(hist_kws={'alpha': .7}, kde_kws={'linewidth': 4}, bins=200)
    print("---- Plotting distribution ----")
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        sns.displot(data, x=property, hue="activity")
        plt.legend()
        plt.title(property)
        if save:
            plt.savefig(saving_dir + '/distribution_' + property + ".png")
        plt.show()


def create_box_plots(data, properties, save=False, saving_dir = "../reports/figures/box_plot_"):
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        ax = sns.boxenplot(x="activity", y=property, hue="activity", data=data)
        plt.legend()
        plt.title(property)
        if save:
            plt.savefig(saving_dir + '/box_plot_' + property + ".png")
        plt.show()


def create_iqr_hist(data, properties, save=False, saving_dir="../reports/figures/iqr_hist_"):
    print("---- Plotting IQR histogram ----")
    kwargs = dict(hist_kws={'alpha': .7}, kde_kws={'linewidth': 4}, bins=100)
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        sns.displot(data, x=property, hue="activity")

        Q1 = 999
        Q3 = -999
        for activity in data.activity:
            subset = data[data["activity"] == activity]
            Q1_subset = np.percentile(subset[property], 25, interpolation='midpoint')
            Q3_subset = np.percentile(subset[property], 75, interpolation='midpoint')
            if Q1_subset < Q1:
                Q1 = Q1_subset
            if Q3_subset > Q3:
                Q3 = Q3_subset

        plt.xlim([Q1, Q3])
        plt.legend()
        plt.title(property)
        if save:
            plt.savefig(saving_dir + '/iqr_hist_' + property + ".png")
        plt.show()


def create_aa_propensity_boxplot(data, save=False, saving_dir="../reports/figures/aa_propensity_"):
    print("---- Creating amino acid propensity plots ----")

    for activity in data.activity.unique():
        print(activity)
        subset = data[data["activity"] == activity]
        df = pd.DataFrame(subset["aa_percentages"].to_list())
        plt.figure(figsize=(10, 7), dpi=80)
        sns.boxplot(data=df * 100)
        plt.ylabel("Amino Acid %")
        plt.title("Amino acid propensity - " + activity)
        if save:
            plt.savefig(saving_dir + '/animo_acid_propensity_' + activity + ".png")
        plt.show()


def create_density_plots(data, properties, save=False, saving_dir="../reports/figures/density_"):
    """
    Very good explanation of density plots: https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

    """
    kwargs = dict(hist_kws={'alpha': .7}, kde_kws={'linewidth': 4}, bins=200)
    print("---- Plotting distribution ----")
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        sns.displot(data, x=property, hue="activity", kind = 'kde')
        plt.legend()
        plt.title(property)
        if save:
            plt.savefig(saving_dir + '/density_' + property + ".png")
        plt.show()
    pass


def create_properties_and_plots(csv_file_with_location_and_activity='src/features/metadata.csv', directory_to_save_properties_file_and_plots='reports/'):
    """
    By default paths are from the root folder: antiviral_peptide_prediction
    headers: path, activity
    you can give absolute paths
    All data should have a column with header Sequence. Does not matter if they have other columns too.
    saving all plots by default

    """
    save_plots = True
    properties_to_plot = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'with_reduced_cysteines', 'with_disulfid_bridges', 'gravy', 'net_charge_at_pH7point4']
    properties_for_box_plot = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'gravy', 'net_charge_at_pH7point4']

    dt = datetime.now().__str__()
    saving_dir = directory_to_save_properties_file_and_plots + dt
    os.mkdir(saving_dir)

    metadata = pd.read_csv(csv_file_with_location_and_activity)
    activities = metadata.shape[0]
    all_data = pd.DataFrame()
    for row in range(activities):
        path = metadata.iloc[row].path
        activity = metadata.iloc[row].activity

        sequences = pd.read_csv(path)
        seq_properties = create_sequence_properties_dataframe(sequences)
        seq_properties['activity'] = activity
        all_data = all_data.append(seq_properties, ignore_index=True)

    all_data.to_csv(saving_dir + '/properties.csv')

    create_box_plots(all_data, properties_for_box_plot, save_plots, saving_dir)

    create_distributions(all_data, properties_to_plot, save_plots, saving_dir)
    create_density_plots(all_data, properties_for_box_plot, save_plots, saving_dir)

    create_iqr_hist(all_data, properties_to_plot, save_plots, saving_dir)

    create_aa_propensity_boxplot(all_data, save_plots, saving_dir)

    return


def get_peptide_composition_each_seq(data_path, kmer):
    sequences = pd.read_csv(data_path)
    # seq_properties = pd.DataFrame(columns=params)
    for seq in sequences.Sequence:
        grams = []
        for i in range(kmer):
            grams.append(zip(*[iter(seq[i:])] * kmer))
        str_ngrams = []
        for ngrams in grams:
            for ngram in ngrams:
                str_ngrams.append("".join(ngram))
        npeptide = pd.Series(str_ngrams).value_counts()
        return npeptide.to_dict()

        # row = pd.DataFrame([[seq, aa_counts, aa_percentages, ]], columns=params)
        # seq_properties = seq_properties.append(row)

def get_peptide_composition_full_file(seq_list, kmer):
    """
    This is occurance of peptide in the full file. Multiple occurances in same peptide are also counted
    """
    str_ngrams = []
    for seq in seq_list:
        grams = []
        for i in range(kmer):
            grams.append(zip(*[iter(seq[i:])] * kmer))
        for ngrams in grams:
            for ngram in ngrams:
                str_ngrams.append("".join(ngram))
    npeptide = pd.Series(str_ngrams).value_counts()
        # print(npeptide)
    print(npeptide)
    return npeptide

def get_peptide_composition_in_number_of_sequences(seq_list, kmer):
    """
       This will return the number of sequences the peptide has occured in.
    """
    unique_kmers_in_peptide = []
    for seq in seq_list:

        temp=[]
        grams = []
        for i in range(kmer):
            grams.append(zip(*[iter(seq[i:])] * kmer)) # all kmers in the sequence has been made
        for ngrams in grams:
            for ngram in ngrams:
                temp.append("".join(ngram)) #flattening it out
        unique_kmers_in_peptide.append(list(set(temp)))
    flattened = [val for sublist in unique_kmers_in_peptide for val in sublist]
    npeptide = pd.Series(flattened).value_counts()
    print(npeptide)
    return npeptide


if __name__ == '__main__':
    create_properties_and_plots(
        '/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan/src/features/metadata.csv',
        '/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan/reports/')


# if __name__ == '__main__':
#
#     """
#     !! Can give absolute paths as follows: !!
#
#     create_properties_and_plots('/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan/src/features/metadata.csv', '/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan/reports/')
#     """
#     # Function by default assumes you are int he root directory: antiviral-peptide-predictions-using-gan.
#     # You can change your current working directory by: os.chdir('<your directory here>')
#
#     # TO generate random sequences:
#     if False:
#         a = generate_random_sequences()
#         a.to_csv('data/generated/generated_random_seq.csv', index=False)
#
#     # create_properties_and_plots('metadata.csv', '../../reports/')
#
#
#     ##### Change this ####
#     calculate_peptide_composition = False
#     calculate_peptide_composition_in_number_of_sequences = True
#     kmer = 5
#     # column = 'Sequence'  # amino_acid_residues; kd_hydrophobicitya
#     # column = 'amino_acid_residues'
#     column = 'kd_hydrophobicitya'
#
#     # file = "AMP_nonAVP_filtered_negative_reduction_combined.csv"
#     # file = "AVPpred_Low_MIC_data_filtered_90perc_reduction_combined.csv"
#     file = "RA_data_negative_data_cdhit_filter_reduction_combined.csv"
#
#     input_file = f"new_seq/{file}"
#     # -----------
#
#     sequences = pd.read_csv('../../data/raw/' + input_file)
#     list_of_seq = sequences[column]
#
#     if calculate_peptide_composition:
#         # get_peptide_composition_each_seq('/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan/data/raw/AVP_data.csv',kmer)
#
#         output_file = str(kmer) + "_" + column + "_" + file
#         npeptide = get_peptide_composition_full_file(list_of_seq, kmer)
#
#         a = npeptide.reset_index()
#         a.to_csv("../../reports/csv_files/%s" % output_file)
#
#     if calculate_peptide_composition_in_number_of_sequences:
#         output_file = str(kmer) + "_" + column + "_" + "num_seq" + "_" + file
#         npeptide = get_peptide_composition_in_number_of_sequences(list_of_seq, kmer)
#         a = npeptide.reset_index()
#         a.to_csv("../../reports/csv_files/%s" % output_file)
#
#         # b = a.to_dict()
#         #
#         # plt.bar(b.keys(), b.values(), color='g')
#         # plt.ylabel("Count of AA in the file")
#         # plt.title("Peptide")
#         # plt.show()
#         #
#         #
#         #
#         # plt.bar(a.iloc[:, 0], a.iloc[:, 1], color='g')
#         # plt.ylabel("Count of AA in the file")
#         # plt.title("Peptide")
#         # plt.show()
#         #
#         # pd.DataFrame.from_dict(npeptide)



