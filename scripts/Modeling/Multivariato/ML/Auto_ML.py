
# LIBs
import os
#
# Custom LIBs
from cust_libs.modeling import *
from cust_libs.misc import Train_Data_Split
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
acc_fldr = 'data/processed/Corsini2021/acc'
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = Train_Data_Split(acc_fldr, 0.8, False, 1811)
    model, attributes = HyperOpt(x_train, y_train, x_test, y_test, '*', '*', 'MSE', 300, 20)
    Save_ML_Model(model, attributes, 'models/multivariate/ML')
