"""
##### DATA PRE-PROCESSING V2 #####
Questo script procede al preprocessamento dei dati:
    - Aggiunge le colonne Pwr_TOT, Pwr_TOT_rel, grad_Pwr_TOT_rel,  Pwr_in, Rendiment
    - Procede al flag delle diverse accendionicon la colonna noACC
    - Suddivide le accensioni in diversi file .csv in /Acc
    - Pulisce le accensioni che non superano la soglia minima di tempo o potenza
"""

# LIBs
from pathlib import Path
import os
#
# Custom LIBs
from cust_libs.data_processing import terna_preprox
from cust_libs.data_processing import preprox
from cust_libs.data_processing import split_ignitions
from cust_libs.data_processing import remove_zeroes
#

# OPTIONS

raw_plant_file = 'data/raw/Corsini2021_Raw.csv'
raw_terna_file = 'data/origin/Terna/Terna_2021_North.xlsx'

PG = 280
PV = 130
Pot_Cal_Met = 36400
min_tresh = 0.01

processed_folder_out = 'data/processed'
#

processed_terna = processed_folder_out + '/terna_Load'
filename = Path(raw_plant_file).stem
plantname = filename.replace('_Raw', '')
processed_plant = processed_folder_out + '/' + plantname + '/' + plantname + '_Processed.csv'
ign_folder_out = processed_folder_out + '/' + plantname + '/acc'
zero_folder_out = processed_folder_out + '/' + plantname
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
terna_preprox(raw_terna_file, processed_folder_out)
preprox(raw_plant_file, PG, PV, processed_terna, processed_folder_out, True,  min_tresh, Pot_Cal_Met, False, False)
split_ignitions(processed_plant, ign_folder_out)
remove_zeroes(processed_plant, zero_folder_out)
