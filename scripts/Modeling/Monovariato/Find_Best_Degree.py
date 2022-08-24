#LIBs
import os

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from cust_libs.modeling import RMSE, MAE
from cust_libs.data_processing import filter_data
import matplotlib.pyplot as plt
import matplotlib as mpl

# Param
max_poly_deg = 20
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_path = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'
data = pd.read_csv(data_path)
data_filt = filter_data(data, None, None, 0.45, 1)                        # > 0.45
x = data['PwrTOT_rel'].values.reshape(-1, 1)
y = data['Rendimento'].values.reshape(-1, 1)
x_filt = data_filt['PwrTOT_rel'].values.reshape(-1, 1)
y_filt = data_filt['Rendimento'].values.reshape(-1, 1)

err_rmse = []
err_mae = []

err_rmse_hp = []
err_mae_hp = []

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

ax.plot(err_rmse, label='RMSE')
ax.plot(err_mae, label='MAE')
ax.plot(err_rmse_hp, label='RMSE 2')
ax.plot(err_mae_hp, label='MAE 2')
ax.set_xlabel(r'Grado polinomio')
ax.set_yscale('log')

plt.legend()
