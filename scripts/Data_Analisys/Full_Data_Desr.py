# LIBs
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# Custom LIBs
from cust_libs.data_processing import filter_data
#
data_path_RAW = 'data/raw/Corsini2021_Raw.csv'
data_path_processed = 'data/processed/Corsini2021/Corsini2021_Processed.csv'
data_path_ON = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_RAW = pd.read_csv(data_path_RAW)
data_PROX = pd.read_csv(data_path_processed)
data_ON = pd.read_csv(data_path_ON)
#
tot_met = sum(data_RAW['Port_Met'].values) / 60         # m^3
tot_energy_in = sum(data_PROX['Pwr_in'].values) / 60    # MWh
tot_energy_out = sum(data_PROX['PwrTOT'].values) / 60   # MWh
rendimento_medio = tot_energy_out / tot_energy_in       # np
rendimento_medio_t = np.average(data_ON['Rendimento'].values)
rendimento_max = max(data_ON['Rendimento'].values)

data_ON['Rendimento'].describe()
data_ON['PwrTOT_rel'].describe()
data_ON['Grad_PwrTOT_rel'].describe() * 100

