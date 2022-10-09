# LIBs
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# Custom LIBs
from cust_libs.misc import transf_fun

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_path_ON = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'

data = pd.read_csv(data_path_ON)

rel2tot, tot2rel = transf_fun(data)

mpl.rcParams["font.size"] = 22
fig, ax = plt.subplots(figsize=(18, 6))

feature = 'Grad_PwrTOT_rel'
censura = True
rend_max = 0.545

if feature == 'PwrTOT_rel':
    X = data[feature].values
    ax.violinplot(X, showmeans=True, vert=False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Potenza Relativa [%]')
    secax_x = ax.secondary_xaxis('top', functions=(rel2tot, tot2rel))
    secax_x.set_xlabel(r'$Potenza\ [MW]$')
    ax.set_xlim([0, 0.9])
    locs_x = ax.get_xticks()
    ax.set_xticks(locs_x, np.round(locs_x * 100, 1))
    plt.tight_layout()
    plt.savefig("IO/baffo_PwrTOT_rel.png", bbox_inches='tight', dpi=300)

elif feature == 'Rendimento':
    X = data[feature].values
    ax.violinplot(X, showmeans=True, vert=False)
    ax.get_yaxis().set_visible(False)
    if censura:
        ax.xaxis.set_ticks(np.linspace(0, rend_max, 6, endpoint=True))
        locs_x = ax.get_xticks()
        ax.set_xticks(locs_x, np.round(locs_x * 100 / rend_max, 1))
        ax.set_xlabel('Rendimento Rel. [%]')

    else:
        locs_x = ax.get_xticks()
        ax.set_xticks(locs_x, np.round(locs_x * 100, 1))
        ax.set_xlabel('Rendimento [%]')
    plt.tight_layout()
    plt.savefig("IO/baffo_Rendimento.png", bbox_inches='tight', dpi=300)
    # plt.show()
elif feature == 'Grad_PwrTOT_rel':
    X = data[feature].values
    ax.violinplot(X, showmeans=True, vert=False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Gradiente [% / min]')
    secax_x = ax.secondary_xaxis('top', functions=(rel2tot, tot2rel))
    secax_x.set_xlabel('Gradiente [MW / min]')
    locs_x = ax.get_xticks()
    ax.set_xticks(locs_x, np.round(locs_x * 100, 1))
    plt.tight_layout()
    plt.savefig("IO/baffo_Gradiente.png", bbox_inches='tight', dpi=300)
