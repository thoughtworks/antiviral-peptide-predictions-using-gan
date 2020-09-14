import tkinter as Tk
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from tkinter import filedialog

try:
    import joypy
except ModuleNotFoundError:
    print("Joypy not installed....")
    print("Installing it now.....")
    import pip

    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

root = Tk.Tk()
root.title("AA Propensity")
root.geometry("400x200")

def dipeptide_encoding(seq, n):
    """
    Returns n-Gram Motif frequency
    https://www.biorxiv.org/content/10.1101/170407v1.full.pdf
    """
    aa_list = list(seq)
    return {''.join(aa_list): n for aa_list, n in Counter(zip(*[aa_list[i:] for i in range(n)])).items() if
            not aa_list[0][-1] == (',')}

def get_cmap():
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist = cmaplist[1:]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    return cmap

def get_filename():
    root.filename = filedialog.askopenfilename(title="Select a CSV file")#, filetypes = (("CSV files", "*.csv")))
    return root.filename


def graph():
    fname = get_filename()
    data = pd.read_csv(fname)
    to_drop = [i for i, s in enumerate(data.Sequence) if ' ' in s]
    data = data.drop(to_drop, axis=0)
    seq_vec = data.Sequence.apply(lambda x: dipeptide_encoding(x, 1)).to_list()
    df = pd.DataFrame(seq_vec)
    df = df.fillna(0)
    
    cmap = get_cmap()
    df = df.sort_index(axis=1)

    fig, axes = joypy.joyplot(df, column=list(df.columns), figsize=(8, 8), fade=True, colormap=cmap,
                          x_range=range(int(df.quantile(0.90).max())), grid=True, ylabelsize=15)
    plt.show()


fm = Tk.Frame(root)

button_group = Tk.Frame(fm)
Tk.Button(fm, text='Plot', width=7,
          command=graph).pack(side=Tk.BOTTOM)
button_group.pack(side=Tk.TOP)
        
fm.pack(side=Tk.TOP)

root.mainloop()
