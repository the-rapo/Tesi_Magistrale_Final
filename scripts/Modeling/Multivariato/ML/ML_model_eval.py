import pandas as pd

from cust_libs.modeling import Load_ML_Model, Model_Plot, RMSE_MAE_plot, Pred_Plot
import matplotlib.pyplot as plt
import os
import sys

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')


Model_type = 'RF'
save_dir = 'temp'
acc = 'data/processed/Corsini2021/acc/Corsini2021_Acc01.csv'
if Model_type == 'RF':
    modelML_path = 'models/multivariate/ML/RND_FOR/RND_FOR_02.joblib'
elif Model_type == 'ALL':
    modelML_path = 'models/multivariate/ML/ALL/ALL_02.joblib'
elif Model_type == 'MLP':
    modelML_path = 'models/multivariate/ML/MLP/MLP_02.joblib'
elif Model_type == 'SVR':
    modelML_path = 'models/multivariate/ML/SVR/SVR_01.joblib'
else:
    sys.exit("Errore")

save_path_error = save_dir + '/' + Model_type + '_error.png'
save_path_model = save_dir + '/' + Model_type + '_model.png'
save_path_example = save_dir + '/' + Model_type + '_example.png'
model, _ = Load_ML_Model(modelML_path)
print(model.best_model())
# Model_Plot(model, n=300, save=save_path_model)
# RMSE_MAE_plot(model, 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv', add_text=False, save=save_path_error)

data_acc = pd.read_csv(acc)
x_pred = data_acc[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
y_true = data_acc['Rendimento'].values.reshape(-1, 1)
Pred_Plot(model, x_pred, y_true, x_transform=None, modelname=None, parameters=None, save=save_path_example)
