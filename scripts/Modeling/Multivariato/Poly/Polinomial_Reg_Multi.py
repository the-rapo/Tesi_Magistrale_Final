# LIBs
import os

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from cust_libs.modeling import Save_Poly_Model, Model_Plot, RMSE_MAE_plot
import numpy as np
from cust_libs.misc import transf_fun
import matplotlib.pyplot as plt
# Param
poly_deg = 5
filtered = False
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_path = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'
data = pd.read_csv(data_path)
x = data[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
y = data['Rendimento'].values.reshape(-1, 1)

poly_trans = PolynomialFeatures(degree=poly_deg, include_bias=False)
x_trans = poly_trans.fit_transform(x)

model = LinearRegression()
model.fit(x_trans, y)
# Save_Poly_Model(model, poly_trans, 'models/multivariate/Poly/deg4')
RMSE_MAE_plot(model, data_path, x_transform=poly_trans, Single_Param=False)
Model_Plot(model, modelname=None, n=100, parameters=None, x_transform=poly_trans, scatter=True)
