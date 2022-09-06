#LIBs
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from cust_libs.modeling import RMSE, MAE
from cust_libs.data_processing import filter_data
import matplotlib.pyplot as plt
import matplotlib as mpl

# Param
max_poly_deg = 20
save_path = 'temp/'

#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_path = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'
data = pd.read_csv(data_path)
data_filt = filter_data(data, None, None, 0.45, 1)                        # > 0.45
x = data[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
y = data['Rendimento'].values.reshape(-1, 1)
x_filt = data_filt[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
y_filt = data_filt['Rendimento'].values.reshape(-1, 1)

err_rmse = []
err_mae = []

err_rmse_hp = []
err_mae_hp = []

mpl.rcParams["font.size"] = 18

for degree in range(1, max_poly_deg + 1, 1):
    poly_trans = PolynomialFeatures(degree=degree, include_bias=False)
    x_trans = poly_trans.fit_transform(x)

    model = LinearRegression()
    model.fit(x_trans, y)
    err_rmse.append(RMSE(model, x, y, x_transform=poly_trans))
    err_mae.append(MAE(model, x, y, x_transform=poly_trans))
    err_rmse_hp.append(RMSE(model, x_filt, y_filt, x_transform=poly_trans))
    err_mae_hp.append(MAE(model, x_filt, y_filt, x_transform=poly_trans))

fig, ax = plt.subplots(figsize=(18, 9))
x = np.linspace(1, 20, 20, endpoint=True)
ax.plot(x, err_rmse, label='RMSE', linewidth=3, marker='o', color='r')
ax.plot(x, err_mae, label='MAE', linewidth=3, marker='^', color='b')
ax.plot(x, err_rmse_hp, label='RMSE (P > 0.45)', linewidth=3, marker='o', linestyle='dashed', alpha=0.5, color='r')
ax.plot(x, err_mae_hp, label='MAE (P > 0.45)', linewidth=3, marker='^', linestyle='dashed', alpha=0.5, color='b')
ax.tick_params(axis='y', which='minor', left=True)
ax.grid(color='black', which='major', axis='y', alpha=0.5, linestyle='dashed', linewidth=1)
ax.grid(color='black', which='minor', axis='y', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax.xaxis.set_ticks(np.arange(1, 21, 4))
ax.set_xlabel(r'Grado polinomio')
ax.set_ylabel('Errore')
ax.set_yscale('log')


plt.legend()
plt.savefig(save_path + 'gradi_pol_errore_multi.png', bbox_inches='tight')