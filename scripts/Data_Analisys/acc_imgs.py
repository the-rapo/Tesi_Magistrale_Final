"""
##### ACC IMG GEN V2 #####
Questo script procede alla creazione di grafici descrittive per le singole accensioni.
Le immagini vengono salvate nella cartella acc/img.
I files hanno nome xxx_ACC_XX_descr.png
"""
# LIBs
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from alive_progress import alive_bar

# Custom LIBs
from cust_libs.misc import folder_tree, mk_dir
from cust_libs.data_processing import transf_fun, filter_data

#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
acc_folder = 'data/processed/Corsini2021/acc'
out_fldr = acc_folder + '/img'
low_P = 0.60
high_P = 0.75

files = folder_tree(acc_folder)

with alive_bar(int(len(files)), force_tty=True) as bar:
    for acc in files:
        data = pd.read_csv(acc)
        filename = Path(acc).stem

        data['DateTime'] = pd.to_datetime(data['DateTime'])

        rel_pwr = data['PwrTOT_rel'].values
        datetime = data['DateTime']
        avg_relPwr = np.average(rel_pwr)

        fig, ax = plt.subplots(2, 1, figsize=(16, 9))
        rel2tot, tot2rel = transf_fun(data)

        # FIG 1
        ax[0].plot(datetime, rel_pwr)
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d : %H'))
        ax[0].axhline(y=avg_relPwr, color='red', linestyle=(0, (5, 10)), label='avg_PWR')

        # fig.autofmt_xdate()
        for tick in ax[0].get_xticklabels():
            tick.set_rotation(30)

        # FIG 2

        LowP_data = filter_data(data, None, None, 0, low_P)
        HighP_data = filter_data(data, None, None, high_P, 1)
        low_perc = len(LowP_data) / len(data) * 100
        high_perc = len(HighP_data) / len(data) * 100
        text_lowP = 'P < ' + "{:.2f}".format(low_P) + ' = ' "{:.1f}".format(low_perc) + '%'
        text_highP = 'P > ' + "{:.2f}".format(high_P) + ' = ' "{:.1f}".format(high_perc) + '%'
        text = text_lowP + '\n' + text_highP
        # This is  the colormap I'd like to use.
        cm = plt.cm.get_cmap('RdYlBu_r')

        _, bins, patches = ax[1].hist(rel_pwr, bins=200, density=False, range=(0.4, 0.9))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

        start, end = ax[1].get_xlim()
        ax[1].xaxis.set_ticks(np.arange(0.4, 0.9, 0.05))

        locs = ax[1].get_yticks()

        ax[1].set_yticks(locs, np.round(locs / len(rel_pwr) * 100, 1))
        ax[1].set_ylabel('Percentuale [%]')
        ax[1].axvline(x=low_P, color='black', linestyle='dashed', linewidth=1, alpha=0.5)
        ax[1].text(0.64, locs.max() * 0.65, text, fontsize=13, color='blue', ha='left')
        # ax[1].text(0.63, len(data) * 0.05, text_highP, fontsize=13, color='red', ha='left')

        # SAVE
        ax[1].axvline(x=high_P, color='black', linestyle='dashed', linewidth=1, alpha=0.5)
        ax[0].axis('tight')
        ax[1].axis('tight')
        fig.subplots_adjust(top=0.95)
        fig.suptitle(filename)
        fig.tight_layout()
        mk_dir(out_fldr, False)
        plt.savefig(out_fldr + '/' + filename + 'Descr' + '.png', dpi=400)
        bar()
        plt.close('all')

plt.close('all')
print('ACC IMG GEN V2 eseguito correttamente')