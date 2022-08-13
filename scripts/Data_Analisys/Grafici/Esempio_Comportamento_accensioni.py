# LIBs
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as collections

from numpy.polynomial import Polynomial

# Custom LIBs
from cust_libs.data_processing import filter_data, transf_fun

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_accensione = 'data/processed/Corsini2021/acc/Corsini2021_Acc05.csv'

data = pd.read_csv(data_accensione)

mpl.rcParams["font.size"] = 18

rel2tot, tot2rel = transf_fun(data)

data['DateTime'] = pd.to_datetime(data['DateTime'])

x = data.index.values
pwr_rel = data['PwrTOT_rel'].values

avg_pwr = np.average(pwr_rel)

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(x, pwr_rel)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=0, ymax=1, where=pwr_rel > 0.75, facecolor='red', alpha=0.2, label='Alta Potenza')

ax.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=0, ymax=1, where=pwr_rel < 0.6, facecolor='green', alpha=0.2, label='Bassa Potenza')
ax.add_collection(collection)

ax.axhline(y=avg_pwr, xmin=0, xmax=len(data), color='k', linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2.5,
           label='Potenza Media')

ax.set_xlim([0, len(data)])
ax.set_ylim([0, 0.9])
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Potenza Relativa [%]')
ax.set_xlabel('Tempo [min]')
secax_y = ax.secondary_yaxis('right', functions=(rel2tot, tot2rel))
secax_y.set_ylabel(r'Potenza $[MW]$')

plt.legend()

plt.savefig("temp/esempio_comport_05" + ".png", bbox_inches='tight')
