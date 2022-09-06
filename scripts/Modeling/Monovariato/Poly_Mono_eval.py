import pandas as pd

from cust_libs.modeling import Load_Poly_model, Model_Plot, RMSE_MAE_plot, Pred_Plot
import os
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

save_dir = 'temp'
acc = 'data/processed/Corsini2021/acc/Corsini2021_Acc01.csv'
model, transformer = Load_Poly_model('models/univariate/Poly/deg4_2')

save_path_error = save_dir + '/poly_mono_error.png'
save_path_model = save_dir + '/poly_mono_model.png'
save_path_example = save_dir + '/poly_mono_example.png'
Model_Plot(model, x_transform=transformer, n=300, save=save_path_model, Single_Param=True)
RMSE_MAE_plot(model, 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv', add_text=False, save=save_path_error,
              x_transform=transformer, Single_Param=True)

data_acc = pd.read_csv(acc)
x_pred = data_acc['PwrTOT_rel'].values.reshape(-1, 1)
y_true = data_acc['Rendimento'].values.reshape(-1, 1)
Pred_Plot(model, x_pred, y_true, x_transform=transformer, modelname=None, parameters=None, save=save_path_example)