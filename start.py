import pandas as pd
import numpy as np
import math
from numba import jit, njit, vectorize, prange, typed, types
import argparse

import warnings
warnings.filterwarnings("ignore")

from SWkernel import SW_K_Mat, SW_kernel
from SpectrumKernel import spectrum_kernel
from mismatchKernel import mismatchKernel
from WDkernel import WD_kernel
from basicKernels import linear_kernel, rbf_kernel, poly_kernel
from CreateKernelMatrix import compute_Ker_mat
from classifiers import Ridge_Classifier, SVM, log_rg_loss


def split_data(X, y, train_ratio):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, train_ratio*100)
    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]
    return X_train, X_test, y_train, y_test




train_data_1 = pd.read_csv('Xtr0.csv' ) 
train_data_mat_1 = pd.read_csv('Xtr0_mat100.csv',header=None) 
test_data_mat_1 = pd.read_csv('Xte0_mat100.csv',header=None) 
train_labels_1 = pd.read_csv('Ytr0.csv' ) 
test_data_1 = pd.read_csv('Xte0.csv' ) 

train_data_2 = pd.read_csv('Xtr1.csv' )
train_data_mat_2 = pd.read_csv('Xtr1_mat100.csv' ,header=None) 
test_data_mat_2 = pd.read_csv('Xte1_mat100.csv',header=None) 
train_labels_2 = pd.read_csv('Ytr1.csv' ) 
test_data_2 = pd.read_csv('Xte1.csv' ) 


train_data_3 = pd.read_csv('Xtr2.csv') 
train_data_mat_3 = pd.read_csv('Xtr2_mat100.csv' ,header=None) 
test_data_mat_3 = pd.read_csv('Xte2_mat100.csv',header=None) 
train_labels_3 = pd.read_csv('Ytr2.csv') 
test_data_3 = pd.read_csv('Xte2.csv') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['string', 'feature'], default='string')
    parser.add_argument('--classifier', choices=['SVM', 'RIDGE'], default='SVM')
    parser.add_argument('--number_of_samples', help='To be able to try on few samples', default=2000)
    parser.add_argument('--Kernel', help='choose a kernel from {spectrum_kernel, SW_kernel, WD_kernel, mismatchKernel} if you are using strings and from {linear, poly, rbf} otherwise', required=True)
    config, unknown = parser.parse_known_args()
    number_of_samples = min(2000, int(config.number_of_samples))
    print(" -- Configuration -- ")
    print("Data_type :", config.data_type)
    print("Classfier :", config.classifier)
    print("Number_of_samples :", number_of_samples)
    print("Kernel :", config.Kernel)
    print("\n")
        
    
    problem = False
    check = (config.Kernel in ['spectrum_kernel', 'SW_kernel', 'WD_kernel', 'mismatchKernel']) or (config.Kernel in ['linear','poly','rbf'])
    if check==False:
        print("Conflict between kernel and data_type, \nplease choose a kernel from {spectrum_kernel, LA_kernel, WD_kernel, mismatchKernel} if you are using strings \nand from {linear,poly,rbf} otherwise")
    
    else:
        if config.data_type=='feature':
            
            ### Train
            X1 = train_data_mat_1[0].str.split(' ').values
            for i , lst in enumerate(X1):
              X1[i] = np.array([float(x) for x  in lst])
            X1 = np.vstack(X1)[:number_of_samples]
            Y1 = train_labels_1['Bound'].to_numpy()[:number_of_samples]
            
            X2 = train_data_mat_2[0].str.split(' ').values
            for i , lst in enumerate(X2):
              X2[i] = np.array([float(x) for x  in lst])
            X2 = np.vstack(X2)[:number_of_samples]
            Y2 = train_labels_2['Bound'].to_numpy()[:number_of_samples]
            
            X3 = train_data_mat_3[0].str.split(' ').values
            for i , lst in enumerate(X3):
              X3[i] = np.array([float(x) for x  in lst])
            X3= np.vstack(X3)[:number_of_samples]
            Y3 = train_labels_3['Bound'].to_numpy()[:number_of_samples]
            
            ### Test 
            X1_test = test_data_mat_1[0].str.split(' ').values
            for i , lst in enumerate(X1_test):
              X1_test[i] = np.array([float(x) for x  in lst])
            X1_test = np.vstack(X1_test)
            
            X2_test = test_data_mat_2[0].str.split(' ').values
            for i , lst in enumerate(X2_test):
              X2_test[i] = np.array([float(x) for x  in lst])
            X2_test = np.vstack(X2_test)
            
            X3_test = test_data_mat_3[0].str.split(' ').values
            for i , lst in enumerate(X3_test):
              X3_test[i] = np.array([float(x) for x  in lst])
            X3_test= np.vstack(X3_test)
            
            
            
            if config.Kernel == 'linear':
                print("-- This will take few milliseconds per dataset to compute the kernel matrix --\n")
                if config.classifier =='SVM': 
                    classifier = SVM(kernel_name=config.Kernel, kernel=linear_kernel, C=20)
                elif config.classifier =='RIDGE':
                    classifier = Ridge_Classifier(lam = 1e-8, kernel_name=config.Kernel, kernel=linear_kernel, loss_func=log_rg_loss)
                
            elif config.Kernel == 'rbf':
                print("-- This will take few milliseconds per dataset to compute the kernel matrix --\n")
                if config.classifier =='SVM': 
                    classifier = SVM(kernel_name=config.Kernel, kernel=rbf_kernel, C=20)
                elif config.classifier =='RIDGE':
                    classifier = Ridge_Classifier(lam = 1e-8, kernel_name=config.Kernel, kernel=rbf_kernel, loss_func=log_rg_loss)
            else:
                classifier = None
                print("Kernel not found")
            
            
            if classifier!=None:
                clf = classifier
                X_train, X_val, Y_train, Y_val = split_data(X1, Y1, train_ratio=0.85)
                clf.fit(X_train, Y_train)
                pred_val , pred_train = clf.predict(X_val, predict_train=True)
                print("Training accuracy for dataset 1 :", np.mean(pred_train==Y_train))
                print("Validation accuracy for dataset 1 :", np.mean(pred_val==Y_val))
                print("testing ...")
                print()
                y_test_1 = clf.predict(X1_test)
                
                clf = classifier
                X_train, X_val, Y_train, Y_val = split_data(X2, Y2, train_ratio=0.85)
                clf.fit(X_train, Y_train)
                pred_val , pred_train = clf.predict(X_val, predict_train=True)
                print("Training accuracy for dataset 2 :", np.mean(pred_train==Y_train))
                print("Validation accuracy for dataset 2 :", np.mean(pred_val==Y_val))
                print("testing ...")
                print()
                y_test_2 = clf.predict(X2_test)
                
                clf = classifier
                X_train, X_val, Y_train, Y_val = split_data(X3, Y3, train_ratio=0.85)
                clf.fit(X_train, Y_train)
                pred_val , pred_train = clf.predict(X_val, predict_train=True)
                print("Training accuracy for dataset 3 :", np.mean(pred_train==Y_train))
                print("Validation accuracy for dataset 3 :", np.mean(pred_val==Y_val))
                print("testing ...")
                print()
                y_test_3 = clf.predict(X3_test)
                                                                                                                  
                    
                
        elif config.data_type=='string': 
            ### Train
            X1 = train_data_1['seq'].to_numpy()[:number_of_samples]
            Y1 = train_labels_1['Bound'].to_numpy()[:number_of_samples]
            
            X2 = train_data_2['seq'].to_numpy()[:number_of_samples]
            Y2 = train_labels_2['Bound'].to_numpy()[:number_of_samples]
            
            X3 = train_data_3['seq'].to_numpy()[:number_of_samples]
            Y3 = train_labels_3['Bound'].to_numpy()[:number_of_samples]
            
            ### Test 
            X1_test = test_data_1['seq'].to_numpy()
            X2_test = test_data_2['seq'].to_numpy()
            X3_test = test_data_3['seq'].to_numpy()
            
            
            if config.Kernel == 'spectrum_kernel':
                if config.classifier =='SVM': 
                    print("-- This will take approximately {0:.0f}min {1:02.0f}s per dataset to compute the kernel matrix --\n".format( *divmod(9.4*(number_of_samples/100)**2, 60)))
                    classifier = SVM(kernel_name =config.Kernel, kernel=spectrum_kernel, spectrum_size=7, C=10)
                elif config.classifier =='RIDGE':
                    classifier = Ridge_Classifier(lam = 1e-8, kernel_name=config.Kernel, kernel=spectrum_kernel, spectrum_size=7, loss_func=log_rg_loss)
            
            elif config.Kernel == 'WD_kernel':
                print("-- This will take approximately {0:.0f}min {1:02.0f}s per dataset to compute the kernel matrix --\n".format( *divmod(3.5*(number_of_samples/500)**2, 60)))
                if config.classifier =='SVM': 
                    classifier = SVM(kernel_name =config.Kernel, kernel=WD_kernel, d=6, C=4)
                elif config.classifier =='RIDGE':
                    classifier = Ridge_Classifier(lam = 1e-8, kernel_name=config.Kernel, kernel=WD_kernel, d=6, loss_func=log_rg_loss)
            
            elif config.Kernel == 'SW_kernel':
                print("-- This will take approximately {0:.0f}min {1:02.0f}s per dataset to compute the kernel matrix --\n".format( *divmod(8.5*(number_of_samples/100)**2, 60)))
                if config.classifier =='SVM': 
                    classifier = SVM(kernel_name =config.Kernel, kernel=SW_kernel, C=0.5)
                elif config.classifier =='RIDGE':
                    classifier = Ridge_Classifier(lam = 1e-8, kernel_name=config.Kernel, kernel=SW_kernel, loss_func=log_rg_loss)
            
            elif config.Kernel == 'mismatchKernel':
                print("-- This will take approximately {0:.0f}min {1:02.0f}s per dataset to compute the kernel matrix --\n".format( *divmod(4.5*(number_of_samples/50)**2, 60)))
                if config.classifier =='SVM': 
                    classifier = SVM(kernel_name=config.Kernel, kernel=mismatchKernel,  C=0.1, m = 1, size = 4)
                elif config.classifier =='RIDGE':
                    classifier = Ridge_Classifier(lam = 1e-8, kernel_name=config.Kernel, kernel=mismatchKernel, m=1, size=4, loss_func=log_rg_loss)
            else:
                classifier = None
                print("Kernel not found")
                
                
            if classifier!=None:
                clf = classifier
                X_train, X_val, Y_train, Y_val = split_data(X1, Y1, train_ratio=0.85)
                clf.fit(X_train, Y_train)
                pred_val , pred_train = clf.predict(X_val, predict_train=True)
                print("Training accuracy for dataset 1 :", np.mean(pred_train==Y_train))
                print("Validation accuracy for dataset 1 :", np.mean(pred_val==Y_val))
                print("testing ...")
                print()
                y_test_1 = clf.predict(X1_test)
                
                clf = classifier
                X_train, X_val, Y_train, Y_val = split_data(X2, Y2, train_ratio=0.85)
                clf.fit(X_train, Y_train)
                pred_val , pred_train = clf.predict(X_val, predict_train=True)
                print("Training accuracy for dataset 2 :", np.mean(pred_train==Y_train))
                print("Validation accuracy for dataset 2 :", np.mean(pred_val==Y_val))
                print("testing ...")
                print()
                y_test_2 = clf.predict(X2_test)
                
                clf = classifier
                X_train, X_val, Y_train, Y_val = split_data(X3, Y3, train_ratio=0.85)
                clf.fit(X_train, Y_train)
                pred_val , pred_train = clf.predict(X_val, predict_train=True)
                print("Training accuracy for dataset 3 :", np.mean(pred_train==Y_train))
                print("Validation accuracy for dataset 3 :", np.mean(pred_val==Y_val))
                print("testing ...")
                print()
                y_test_3 = clf.predict(X3_test)
                
                                                                                              
            
        if problem == False:
            y_test = y_test_1 + y_test_2 + y_test_3
            
            ### Create Test file
            output_file = open('Yte.csv', "w")
            output_file.write("Id,Bound\n")
            for i in range(3000):
              output_file.write("%s,%d\n" % (i, y_test[i]))
            output_file.close()
            print("Succesfully wrote Yte.csv'")
        
              
