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
# Params
setting = 'Read'
acc_path = 'data/processed/Corsini2021/acc'
n_accensione = 1
#
poi=[
    [1, 1, 1, 1],
    [2, 2, 2, 2]
    ]
#
if setting == 'Read':
    acc = acc_path + '/Corsini2021_Acc' + str(n_accensione).zfill(2) + '.csv'
    data = pd.read_csv(acc)
    data['PwrTOT_rel'].plot()
    plt.show()

