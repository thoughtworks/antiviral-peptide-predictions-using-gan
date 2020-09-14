import pandas as pd
# import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis

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
        gravy = X.gravy()
        flexibility = X.flexibility()
        # X.protein_scale()
        net_charge_at_pH7 = X.charge_at_pH(7)

        row = pd.DataFrame([[seq, aa_counts, aa_percentages, molecular_weight, aromaticity, instability_index,
                             isoelectric_point, sec_struc, helix, turn, sheet, epsilon_prot, with_reduced_cysteines,
                             with_disulfid_bridges, gravy, flexibility]], columns=params)
        seq_properties = seq_properties.append(row)
    return seq_properties

sequences = pd.read_csv('../data/raw/avp_sequences.csv')
avp_seq_properties = create_sequence_properties_dataframe(sequences)

# ---- Visualizations on seq properties ----


kwargs = dict(hist_kws={'alpha': .3}, kde_kws={'linewidth': 4}, bins=200)

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.molecular_weight, color="g", label="Real", **kwargs)
plt.legend()
plt.title("molecular_weight")
plt.savefig("../reports/figures/molecular_weight.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.aromaticity, color="g", label="Real", **kwargs)
plt.legend()
plt.title("aromaticity")
plt.savefig("../reports/figures/aromaticity.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.instability_index, color="g", label="Real", **kwargs)
plt.legend()
plt.title("instability_index")
plt.savefig("../reports/figures/instability_index.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.isoelectric_point, color="g", label="Real", **kwargs)
# sns.distplot(avp_seq_properties.isoelectric_point, color="r", label="Generated", **kwargs)
plt.legend()
# plt.xlim(-0.25,0.75)
plt.title("isoelectric_point")
plt.savefig("../reports/figures/isoelectric_point.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.helix, color="g", label="Real", **kwargs)
plt.legend()
plt.title("secondary structure helix")
plt.savefig("../reports/figures/helix.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.turn, color="g", label="Real", **kwargs)
plt.legend()
plt.title("secondary structure turn")
plt.savefig("../reports/figures/turn.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.sheet, color="g", label="Real", **kwargs)
plt.legend()
plt.title("secondary structure sheet")
plt.savefig("../reports/figures/sheet.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.with_reduced_cysteines, color="g", label="Real", **kwargs)
plt.legend()
plt.title("molar_extinction_coefficient  with_reduced_cysteines")
plt.savefig("../reports/figures/with_reduced_cysteines.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.with_disulfid_bridges, color="g", label="Real", **kwargs)
plt.legend()
plt.title("molar_extinction_coefficient  with_disulfid_bridges")
plt.savefig("../reports/figures/with_disulfid_bridges.png")
plt.show()

plt.figure(figsize=(10, 7), dpi=80)
sns.distplot(avp_seq_properties.gravy, color="g", label="Real", **kwargs)
plt.legend()
plt.title("gravy  ")
plt.savefig("../reports/figures/gravy.png")
plt.show()

# Create box plot comparison
