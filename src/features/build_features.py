import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

import os


def create_sequence_properties_dataframe(sequences):
    params = ['seq', 'aa_counts', 'aa_percentages', 'molecular_weight', 'aromaticity', 'instability_index',
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

def create_distributions(amp_data, non_amp_data, generated_avp_data, properties, save=False):
    kwargs = dict(hist_kws={'alpha': .3}, kde_kws={'linewidth': 4}, bins=200)
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        sns.distplot(amp_data[property], color="g", label="Real_AVP", **kwargs)
        sns.distplot(non_amp_data[property], color="y", label="non-AVP", **kwargs)
        sns.distplot(generated_avp_data[property], color="m", label="generated-AVP", **kwargs)
        plt.legend()
        plt.title(property)
        if save:
            plt.savefig("../reports/figures/distribution_"+property+".png")
        plt.show()


def create_box_plots(data, properties, save=False):
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        ax = sns.boxenplot(x="activity", y=property, hue="activity", data=data, palette="Set3")
        plt.legend()
        plt.title(property)
        if save:
            plt.savefig("../reports/figures/box_plot_"+property+".png")
        plt.show()


def create_iqr_hist(positive_data, negative_data, generated_data, properties, save=False):
    kwargs = dict(hist_kws={'alpha': .3}, kde_kws={'linewidth': 4}, bins=100)
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        sns.distplot(positive_data[property], hist=True, color="g", label="Real_AMP", **kwargs)
        sns.distplot(negative_data[property], hist=True, color="y", label="non-AMP", **kwargs)
        sns.distplot(generated_data[property], hist=True, color="m", label="non-AMP", **kwargs)

        Q1_amp = np.percentile(positive_data[property], 25, interpolation='midpoint')
        Q3_amp = np.percentile(positive_data[property], 75, interpolation='midpoint')
        Q1_non_amp = np.percentile(negative_data[property], 25, interpolation='midpoint')
        Q3_non_amp = np.percentile(negative_data[property], 75, interpolation='midpoint')
        Q1_generated_amp = np.percentile(generated_data[property], 25, interpolation='midpoint')
        Q3_generated_amp = np.percentile(generated_data[property], 75, interpolation='midpoint')

        plt.xlim([min(Q1_amp,Q1_non_amp,Q1_generated_amp), max(Q3_amp, Q3_non_amp,Q3_generated_amp)])
        plt.legend()
        plt.title(property)
        if save:
            plt.savefig("../reports/figures/iqr_hist_" + property + ".png")
        plt.show()

def create_properties_and_plots(avp_data_path, non_avp_data_path, generated_avp_data_path, save_plots=False):
    """
    params:
    All data must have sequences under the header 'Sequence'
    """
    sequences = pd.read_csv(avp_data_path)
    avp_seq_properties = create_sequence_properties_dataframe(sequences)
    avp_seq_properties['activity'] = 'AVP'

    sequences = pd.read_csv(non_avp_data_path)
    non_avp_seq_properties = create_sequence_properties_dataframe(sequences)
    non_avp_seq_properties['activity'] = 'non-AVP'

    sequences = pd.read_csv(generated_avp_data_path)
    generated_avp_seq_properties = create_sequence_properties_dataframe(sequences)
    generated_avp_seq_properties['activity'] = 'generated-AVP'

    all_data_temp = avp_seq_properties.append(non_avp_seq_properties, ignore_index=True)
    all_data = all_data_temp.append(generated_avp_seq_properties, ignore_index=True)
    del all_data_temp

    properties_to_plot = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn','sheet', 'with_reduced_cysteines', 'with_disulfid_bridges', 'gravy']
    create_distributions(avp_seq_properties, non_avp_seq_properties, generated_avp_seq_properties, properties_to_plot, save_plots)
    properties_for_box_plot = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'gravy']
    create_box_plots(all_data, properties_for_box_plot, save_plots)
    create_iqr_hist(avp_seq_properties, non_avp_seq_properties, generated_avp_seq_properties, properties_to_plot, save_plots)

def dipeptide_encoding(seq, n):
    """
    Returns n-Gram Motif frequency
    https://www.biorxiv.org/content/10.1101/170407v1.full.pdf
    """
    aa_list = list(seq)
    return {''.join(aa_list): n for aa_list, n in Counter(zip(*[aa_list[i:] for i in range(n)])).items() if
            not aa_list[0][-1] == (',')}
    
def create_aa_propensity_boxplot(fname, save=False):
    data = pd.read_csv(fname)
    to_drop = [i for i, s in enumerate(data.Sequence) if ' ' in s]
    data = data.drop(to_drop, axis=0)
    seq_vec = data.Sequence.apply(lambda x: dipeptide_encoding(x, 1)).to_list()
    df = pd.DataFrame(seq_vec)
    df = df.fillna(0)
    df = df.sort_index(axis=1)
    df_fraction = df.div(df.sum(axis=1), axis=0)

    sns.boxplot(data=df_fraction*100)
    plt.ylabel("Amino Acid %")
    plt.title("Amino acid propensity")
    if save:
            plt.savefig("../reports/figures/animo_acid_propensity.png")
    plt.show()

if __name__ == '__main__':
    os.chdir("/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan/src")

    save_plots = True

    avp_sequences = pd.read_csv('../data/raw/AVP_data.csv')
    avp_seq_properties = create_sequence_properties_dataframe(avp_sequences)
    avp_seq_properties['activity'] = 'AVP'
    non_avp_seq_properties = create_sequence_properties_dataframe(pd.read_csv('../data/raw/non_AVP_data.csv'))
    non_avp_seq_properties['activity'] = 'non-AVP'

    generated_avp_seq_properties = create_sequence_properties_dataframe(pd.read_csv('../data/generated/generated_AVP_data.csv'))
    generated_avp_seq_properties['activity'] = 'generated-AVP'

    all_data_temp = avp_seq_properties.append(non_avp_seq_properties, ignore_index=True)
    all_data = all_data_temp.append(generated_avp_seq_properties, ignore_index=True)
    del all_data_temp


    properties_to_plot = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'with_reduced_cysteines', 'with_disulfid_bridges', 'gravy', 'net_charge_at_pH7point4']
    create_distributions(avp_seq_properties, non_avp_seq_properties, generated_avp_seq_properties, properties_to_plot, save_plots)
    properties_for_box_plot = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'gravy', 'net_charge_at_pH7point4']
    create_box_plots(all_data, properties_for_box_plot, save_plots)
    create_iqr_hist(avp_seq_properties, non_avp_seq_properties, generated_avp_seq_properties, properties_to_plot, save_plots)
