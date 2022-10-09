# LIBs
import os

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from cust_libs.modeling import RMSE, MAE, RMSE_MAE_plot, Save_Poly_Model
from cust_libs.data_processing import filter_data
import numpy as np
from cust_libs.misc import transf_fun
import matplotlib.pyplot as plt
# Param
poly_deg = 4
filtered = False
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_path = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'
data = pd.read_csv(data_path)
data = filter_data(data, -0.005, 0.005, None, None)

x = data['PwrTOT_rel'].values.reshape(-1, 1)
y = data['Rendimento'].values.reshape(-1, 1)

poly_trans = PolynomialFeatures(degree=poly_deg, include_bias=True)
x_trans = poly_trans.fit_transform(x)

model = LinearRegression()
model.fit(x_trans, y)

'''
err_rmse = RMSE(model, x, y, x_transform=poly_trans)
err_mae = MAE(model, x, y, x_transform=poly_trans)


rel2tot, tot2rel = transf_fun(data)

x_plot = np.linspace(0.01, 0.95, 100, endpoint=True)
x_plot = x_plot.reshape(-1, 1)
x_plot_2 = poly_trans.transform(x_plot)
y_rend = model.predict(x_plot_2)

fig, ax = plt.subplots(figsize=(18, 9))
ax.scatter(x, y, marker='o')
ax.plot(x_plot, y_rend, color='red', linewidth=4)

ax.set_ylabel(r'Rendimento')
ax.set_xlabel(r'Potenza Relativa')
secax_x = ax.secondary_xaxis(
    'top', functions=(rel2tot, tot2rel))
secax_x.set_xlabel(r'$P\ $ [MW]')

if filtered:
    fig.suptitle(r'Curva di rendimento Impianto Corsini @ $ | \nabla  P | < $ 2  MW/min' + '\n'
                 r'Regressione polinomiale grado ' + str(poly_deg))
else:
    fig.suptitle(r'Curva di rendimento Impianto Corsini' + '\n' + r'Regressione polinomiale grado ' + str(poly_deg))

RMSE_text = 'RMSE = ' "{:.4f}".format(err_rmse)
MAE_text = ' MAE = ' "{:.4f}".format(err_mae)

ax.text(0.7, 0.15, RMSE_text, fontsize='x-large')
ax.text(0.7, 0.10, MAE_text, fontsize='x-large')
plt.show()
'''
Save_Poly_Model(model, poly_trans, 'models/univariate/Poly/deg4_low_2')
RMSE_MAE_plot(model, data_path, x_transform=poly_trans, Single_Param=True)
