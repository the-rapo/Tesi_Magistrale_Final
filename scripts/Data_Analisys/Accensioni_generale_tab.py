# LIBs
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Custom LIBs

#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
# Params
setting = 'Read'
acc_path = 'data/processed/Corsini2021/acc'
n_accensione = 55
AVG = True
#
if AVG:
    t = 0
    potenza_media_tot = 0
    rendimento_medio_tot = 0
    rendimento_medio_t_tot = 0
    for n_accensione in range(1,56,1):
        acc = acc_path + '/Corsini2021_Acc' + str(n_accensione).zfill(2) + '.csv'
        data = pd.read_csv(acc)
        #
        tot_energy_in = sum(data['Pwr_in'].values) / 60  # MWh
        tot_energy_out = sum(data['PwrTOT'].values) / 60  # MWh
        rendimento_medio = tot_energy_out / tot_energy_in  # np
        rendimento_medio_t = np.average(data['Rendimento'].values)
        potenza_media = np.average(data['PwrTOT_rel'].values)
        t += np.round(len(data) / 60, 0)
        potenza_media_tot += potenza_media * 100
        rendimento_medio_tot += rendimento_medio * 100
        rendimento_medio_t_tot += rendimento_medio_t * 100
    print('potenza media avg: ' + str(np.round(potenza_media_tot / 55, 1)))
    print('rendimento medio avg: ' + str(np.round(rendimento_medio_tot / 55, 1)))
    print('rendimento medio t avg: ' + str(np.round(rendimento_medio_t_tot / 55, 1)))
    print('t: ' + str(np.round(t / 55, 1)))
else:
    acc = acc_path + '/Corsini2021_Acc' + str(n_accensione).zfill(2) + '.csv'
    data = pd.read_csv(acc)
    #
    tot_energy_in = sum(data['Pwr_in'].values) / 60    # MWh
    tot_energy_out = sum(data['PwrTOT'].values) / 60   # MWh
    rendimento_medio = tot_energy_out / tot_energy_in       # np
    rendimento_medio_t = np.average(data['Rendimento'].values)
    potenza_media = np.average(data['PwrTOT_rel'].values)
    print('Data: ' + data['DateTime'].iloc[0])
    print('Tempo: ' + str(np.round(len(data) / 60, 0)))

    print('Potenza media: ' + str(np.round(potenza_media * 100, 1)))
    print('Rendimento medio: ' + str(np.round(rendimento_medio * 100, 1)))
    print('Rendimento medio t : ' + str(np.round(rendimento_medio_t * 100, 1)))
    data['PwrTOT_rel'].plot()
    plt.show()
