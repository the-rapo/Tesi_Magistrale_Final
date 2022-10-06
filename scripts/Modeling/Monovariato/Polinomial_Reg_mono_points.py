# LIBs
import os

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from cust_libs.modeling import RMSE, MAE, RMSE_MAE_plot, Save_Poly_Model
import numpy as np
import sys
from cust_libs.misc import transf_fun
import matplotlib.pyplot as plt
# Param
poly_deg = 4
filtered = False
data = 'paper2'
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
if data == 'paper':
    x = [0.482, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]
    y = [0.526, 0.53, 0.543, 0.555, 0.562, 0.57, 0.578, 0.582, 0.588, 0.59, 0.592, 0.593]
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    name = 'paper'
elif data == 'paper2':
    x = [0.4, 0.5, 0.6, 0.70, 0.80, 0.90, 1]
    y = [0.44, 0.481, 0.51, 0.530, 0.5376, 0.546, 0.56]
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    name = 'paper2'
else:
    sys.exit("Errore")
'''
data = pd.read_csv(data_path)
x = data['PwrTOT_rel'].values.reshape(-1, 1)
y = data['Rendimento'].values.reshape(-1, 1)
'''
poly_trans = PolynomialFeatures(degree=poly_deg, include_bias=False)
x_trans = poly_trans.fit_transform(x)

model = LinearRegression()
model.fit(x_trans, y)

Save_Poly_Model(model, poly_trans, 'models/univariate/Poly/' + name)
# RMSE_MAE_plot(model, data_path, x_transform=poly_trans, Single_Param=True)
