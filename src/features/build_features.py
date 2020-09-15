import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

import os

os.chdir("/Users/shraddhasurana/Desktop/projects/E4R/LifeSciences/ddh/antiviral-peptide-predictions-using-gan/src")

def create_sequence_properties_dataframe(sequences):
    params = ['seq', 'aa_counts', 'aa_percentages', 'molecular_weight', 'aromaticity', 'instability_index',
              'isoelectric_point', 'sec_struc', 'helix', 'turn', 'sheet', 'epsilon_prot', 'with_reduced_cysteines',
              'with_disulfid_bridges', 'gravy', 'flexibility']

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
                             with_disulfid_bridges, gravy, flexibility]], columns=params)
        seq_properties = seq_properties.append(row)
    return seq_properties

sequences = pd.read_csv('../data/raw/AVP_data.csv')
avp_seq_properties = create_sequence_properties_dataframe(sequences)
avp_seq_properties['activity'] = 'AVP'
non_avp_seq_properties = create_sequence_properties_dataframe(pd.read_csv('../data/raw/non_AVP_data.csv'))
non_avp_seq_properties['activity'] = 'non-AVP'

all_data = avp_seq_properties.append(non_avp_seq_properties, ignore_index = True)


# ---- Visualizations on seq properties ----

def create_distributions(amp_data, non_amp_data, properties):
    kwargs = dict(hist_kws={'alpha': .3}, kde_kws={'linewidth': 4}, bins=200)
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        sns.distplot(amp_data[property], color="g", label="Real_AMP", **kwargs)
        sns.distplot(non_amp_data[property], color="y", label="non-AMP", **kwargs)
        plt.legend()
        plt.title(property)
        plt.savefig("../reports/figures/distribution_"+property+".png")
        plt.show()


def create_box_plots(data, properties):
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        ax = sns.boxenplot(x="activity", y=property, hue="activity", data=data, palette="Set3")
        plt.legend()
        plt.title(property)
        plt.savefig("../reports/figures/box_plot_"+property+".png")
        plt.show()


def create_iqr_hist(amp_data, non_amp_data, properties):
    kwargs = dict(hist_kws={'alpha': .3}, kde_kws={'linewidth': 4}, bins=100)
    for property in properties:
        print(property)
        plt.figure(figsize=(10, 7), dpi=80)
        sns.distplot(amp_data[property], hist=True, color="g", label="Real_AMP", **kwargs)
        sns.distplot(non_amp_data[property], hist=True, color="y", label="non-AMP", **kwargs)

        Q1_amp = np.percentile(amp_data[property], 25, interpolation='midpoint')
        Q3_amp = np.percentile(amp_data[property], 75, interpolation='midpoint')
        Q1_non_amp = np.percentile(non_amp_data[property], 25, interpolation='midpoint')
        Q3_non_amp = np.percentile(non_amp_data[property], 75, interpolation='midpoint')

        plt.xlim([min(Q1_amp,Q1_non_amp), max(Q3_amp, Q3_non_amp)])
        plt.legend()
        plt.title(property)
        plt.savefig("../reports/figures/iqr_hist_" + property + ".png")
        plt.show()


properties_to_plot = ['molecular_weight', 'aromaticity', 'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet', 'with_reduced_cysteines', 'with_disulfid_bridges', 'gravy']
create_distributions(avp_seq_properties, non_avp_seq_properties, properties_to_plot)
create_box_plots(all_data, properties_to_plot)
create_iqr_hist(avp_seq_properties, non_avp_seq_properties, properties_to_plot)
