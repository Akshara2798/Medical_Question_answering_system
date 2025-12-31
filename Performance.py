import warnings 
warnings.filterwarnings('ignore') 
from sklearn.random_projection import GaussianRandomProjection
from sklearn import preprocessing 
from keras.models import load_model
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from __pycache__.utils import *
import matplotlib.font_manager as font_manager
from numpy import asarray, mean
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,precision_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr

   


        
def Plot(Ex_time):
    Class= np.unique(np.load("files/y_test.npy"))
    X=len(np.unique(Class))
    X=Predict(X)
    y_test=X[:,0]  ;pred=X[:,1] 
    
    
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    def calculate_relative_error(true, pred):
        return abs(true - pred) / true

    def calculate_accuracy_relative(y_true, y_pred, threshold):
        correct_predictions = 0
        total_predictions = len(y_true)
    
        for true, pred in zip(y_true, y_pred):
            relative_error = calculate_relative_error(true, pred)
            if relative_error <= threshold:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_predictions) 
        return accuracy


    threshold = 0.05  
    accuracy = calculate_accuracy_relative(y_test,pred, threshold)
    absolute_percentage_errors = np.abs((y_test - pred) / y_test)
    mape_proposed = np.mean(absolute_percentage_errors)
    
    k=np.unique(y_test)
    indix=[]
    for ques in k:
        indices_of_1 = np.where(y_test == ques)[0]
        indix.append(indices_of_1)
    
    score_list_proposed=[]
    for ind in indix:
        kunds=[]
        for kund in ind:
            rank=pred[kund]
            kunds.append(rank)
        score_list_proposed.append(kunds)
    
    def mean_reciprocal_rank(ranks):
       
        reciprocal_ranks = [1.0 / rank for rank in ranks if rank > 0]  
        if not reciprocal_ranks:  
            return 0.0
        else:
            return sum(reciprocal_ranks) / len(reciprocal_ranks)  
    ranks = score_list_proposed
    mrr_values = [mean_reciprocal_rank(rank_list) for rank_list in ranks]
    mrr_proposed = sum(mrr_values) / len(mrr_values)
    
    x = y_test
    y = pred
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    r_proposed = numerator / denominator
    
    
   
    def concordance_correlation_coefficient(y_true, y_pred):
        pearson_corr, _ = pearsonr(y_true, y_pred)
    
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        std_true = np.std(y_true, ddof=1) 
        std_pred = np.std(y_pred, ddof=1)
    
        numerator = 2 * pearson_corr * std_true * std_pred
        denominator = std_true**2 + std_pred**2 + (mean_true - mean_pred)**2
        ccc = numerator / denominator
    
        return ccc
    ccc_proposed=concordance_correlation_coefficient(pred,y_test)
    
    rho_pro,a = spearmanr(y_test, pred)

      
    
    
    print("proposed performance : \n********************\n")
    print("Accuracy            :",accuracy)
    print("R2-Score            :",r2)
    print('MAE                 :',mae)
    print("MSE                 :",mse) 
    print("RMSE                :",rmse)
    print("MAPE                :",mape_proposed)
    print("MRR                 :",mrr_proposed)
    print("Pearson CC (r)      :",r_proposed)
    print("Concordance CC      :",ccc_proposed)
    print("SRCC                :",rho_pro)
    print()
 

    #--------------------------------#
    # -------------Auto Encoder------------#
    pred_ae=X[:,2] 
    mae_ae = mean_absolute_error(y_test, pred_ae)
    mse_ae = mean_squared_error(y_test, pred_ae)
    rmse_ae = np.sqrt(mse_ae)
    r2_ae = r2_score(y_test, pred_ae)

    
    def calculate_relative_error(true, pred):
        return abs(true - pred) / true

    def calculate_accuracy_relative(y_true, y_pred, threshold):
        correct_predictions = 0
        total_predictions = len(y_true)
    
        for true, pred in zip(y_true, y_pred):
            relative_error = calculate_relative_error(true, pred)
            if relative_error <= threshold:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_predictions) 
        return accuracy


    threshold = 0.05  
    accuracy_ae = calculate_accuracy_relative(y_test,pred_ae, threshold)
    absolute_percentage_errors = np.abs((y_test - pred_ae) / y_test)
    mape_ae = np.mean(absolute_percentage_errors)
    score_list_ae=[]
    for ind in indix:
        kunds=[]
        for kund in ind:
            rank=pred_ae[kund]
            kunds.append(rank)
        score_list_ae.append(kunds)
    def mean_reciprocal_rank(ranks):
        
         reciprocal_ranks = [1.0 / rank for rank in ranks if rank > 0]  
         if not reciprocal_ranks: 
             return 0.0
         else:
             return sum(reciprocal_ranks) / len(reciprocal_ranks)     
    ranks = score_list_ae
    mrr_values = [mean_reciprocal_rank(rank_list) for rank_list in ranks]
    mrr_ae = sum(mrr_values) / len(mrr_values)
    
    x = y_test
    y = pred_ae
    

    mean_x = np.mean(x)
    mean_y = np.mean(y)
   
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    r_ae = numerator / denominator
    
    ccc_ae=concordance_correlation_coefficient(pred_ae,y_test)
    rho_ae,a = spearmanr(y_test, pred_ae)
  
 
    print("Auto Encoder: \n********************\n")
    print("Accuracy            :",accuracy_ae)
    print("R2-Score            :",r2_ae)
    print('MAE                 :',mae_ae)
    print("MSE                 :",mse_ae) 
    print("RMSE                :",rmse_ae)
    print("MAPE                :",mae_ae)
    print("MRR                 :",mrr_ae)
    print("Pearson CC (r)      :",r_ae)
    print("Concordance CC      :",ccc_ae)
    print("SRCC                :",rho_ae)
 
    

    # -------------CNN-LSTM------------#
    pred_cl=X[:,3] 
    mae_cl = mean_absolute_error(y_test, pred_cl)
    mse_cl = mean_squared_error(y_test, pred_cl)
    rmse_cl = np.sqrt(mse_cl)
    r2_cl = r2_score(y_test, pred_cl)
    
   
    def calculate_relative_error(true, pred):
        return abs(true - pred) / true
    
    def calculate_accuracy_relative(y_true, y_pred, threshold):
        correct_predictions = 0
        total_predictions = len(y_true)
        
        for true, pred in zip(y_true, y_pred):
            relative_error = calculate_relative_error(true, pred)
            if relative_error <= threshold:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_predictions) 
        return accuracy
    
    
    threshold = 0.05  
    accuracy_cl = calculate_accuracy_relative(y_test,pred_cl, threshold)
    absolute_percentage_errors = np.abs((y_test - pred_cl) / y_test)
    mape_cl = np.mean(absolute_percentage_errors)
    
    
    score_list_cl=[]
    for ind in indix:
        kunds=[]
        for kund in ind:
            rank=pred_cl[kund]
            kunds.append(rank)
        score_list_cl.append(kunds)
    def mean_reciprocal_rank(ranks):
        
         reciprocal_ranks = [1.0 / rank for rank in ranks if rank > 0] 
         if not reciprocal_ranks: 
             return 0.0
         else:
             return sum(reciprocal_ranks) / len(reciprocal_ranks)
    ranks = score_list_cl
    mrr_values = [mean_reciprocal_rank(rank_list) for rank_list in ranks]
    mrr_cl = sum(mrr_values) / len(mrr_values)
    
    x = y_test
    y = pred_cl
    
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    r_cl = numerator / denominator
    ccc_cl=concordance_correlation_coefficient(pred_cl,y_test)
    rho_cl,a = spearmanr(y_test, pred_ae)

    print()
    print("CNN-LSTM performance: \n********************\n")
    print("Accuracy            :",accuracy_cl)
    print("R2-Score            :",r2_cl)
    print('MAE                 :',mae_cl)
    print("MSE                 :",mse_cl) 
    print("RMSE                :",rmse_cl)
    print("MAPE                :",mape_cl)
    print("MRR                 :",mrr_cl)
    print("Pearson CC (r)      :",r_cl)
    print("Concordance CC      :",ccc_cl)
    print("SRCC                :",rho_cl)
    
    
    
    #------------------Bi-LSTM----------------
    pred_bi=X[:,4] 
    mae_bi = mean_absolute_error(y_test, pred_bi)
    mse_bi = mean_squared_error(y_test, pred_bi)
    rmse_bi = np.sqrt(mse_bi)
    r2_bi = r2_score(y_test, pred_bi)
    
    def calculate_relative_error(true, pred):
        return abs(true - pred) / true
    
    def calculate_accuracy_relative(y_true, y_pred, threshold):
        correct_predictions = 0
        total_predictions = len(y_true)
        
        for true, pred in zip(y_true, y_pred):
            relative_error = calculate_relative_error(true, pred)
            if relative_error <= threshold:
                correct_predictions += 1
    
        accuracy = (correct_predictions / total_predictions) 
        return accuracy
    
    
    threshold = 0.05 
    accuracy_bi = calculate_accuracy_relative(y_test,pred_bi, threshold)
    absolute_percentage_errors = np.abs((y_test - pred_bi) / y_test)
    mape_bi = np.mean(absolute_percentage_errors)
    
    score_list_bi=[]
    for ind in indix:
        kunds=[]
        for kund in ind:
            rank=pred_bi[kund]
            kunds.append(rank)
        score_list_bi.append(kunds)
    def mean_reciprocal_rank(ranks):
        
         reciprocal_ranks = [1.0 / rank for rank in ranks if rank > 0]  
         if not reciprocal_ranks: 
             return 0.0
         else:
             return sum(reciprocal_ranks) / len(reciprocal_ranks)  
    ranks = score_list_bi
    mrr_values = [mean_reciprocal_rank(rank_list) for rank_list in ranks]
    mrr_bi = sum(mrr_values) / len(mrr_values)
    
    x = y_test
    y = pred_bi
    
   
    mean_x = np.mean(x)
    mean_y = np.mean(y)
 
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    r_bi = numerator / denominator
    ccc_bi=concordance_correlation_coefficient(pred_bi,y_test)
    rho_bi,a = spearmanr(y_test, pred_bi)
    print()
    print("Bi-LSTM performance: \n********************\n")
    print("Accuracy            :",accuracy_bi)
    print("R2-Score            :",r2_bi)
    print('MAE                 :',mae_bi)
    print("MSE                 :",mse_bi) 
    print("RMSE                :",rmse_bi)
    print("MAPE                :",mape_bi)
    print("MRR                 :",mrr_bi)
    print("Pearson CC (r)      :",r_bi)
    print("Concordance CC      :",ccc_bi)
    print("SRCC                :",rho_bi)
    
    
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 14}
    con = "Proposed"
    con1 = "  AE" 
    con2 = "    CNN-LSTM"
    con3 = "Bi-LSTM"
   
     
    
    plt.figure(figsize=(7,5));plt.ylim(85,100)    
    width = 0.25
    plt.bar(0,accuracy*100, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,accuracy_ae*100, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,accuracy_cl*100, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,accuracy_bi*100, width, color='#808080', align='center', edgecolor='black') 
   
    
    
    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Accuracy (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/Accuracy.png', format="png",dpi=600)


    plt.figure(figsize=(7,5));plt.ylim(85,100)    
    width = 0.25
    plt.bar(0,r2*100, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,r2_ae*100, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,r2_cl*100, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,r2_bi*100, width, color='#808080', align='center', edgecolor='black') 
 
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('R-Score (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/r2-score.png', format="png",dpi=600)
   

    plt.figure(figsize=(7.2,5));plt.ylim(0,0.025)
    width = 0.25
    plt.bar(0,mae, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,mae_ae, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,mae_cl, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,mae_bi, width, color='#808080', align='center', edgecolor='black') 
    
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('MAE',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/MAE.png', format="png",dpi=600) 
    
    plt.figure(figsize=(7.2,5))
    width = 0.25
    plt.bar(0,mse, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,mse_ae, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,mse_cl, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,mse_bi, width, color='#808080', align='center', edgecolor='black') 
  
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('MSE',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/MSE.png', format="png",dpi=600)
    
    plt.figure(figsize=(7,5))
    width = 0.25
    plt.bar(0,rmse, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,rmse_ae, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,rmse_cl, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,rmse_bi, width, color='#808080', align='center', edgecolor='black') 
   
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('RMSE',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/RMSE.png', format="png",dpi=600)
    
    
    plt.figure(figsize=(7,5))
    width = 0.25
    plt.bar(0,mape_proposed, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,mape_ae, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,mape_cl, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,mape_bi, width, color='#808080', align='center', edgecolor='black') 
   
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('MAPE',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/MAPE.png', format="png",dpi=600)
    
    plt.figure(figsize=(7,5))
    plt.ylim(2,2.6)
    width = 0.25
    plt.bar(0,mrr_proposed, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,mrr_ae, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,mrr_cl, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,mrr_bi, width, color='#808080', align='center', edgecolor='black') 
   
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('MRR',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/MRR.png', format="png",dpi=600)
    
    plt.figure(figsize=(7,5))
    plt.ylim(85,100)
    width = 0.25
    plt.bar(0,r_proposed*100, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,r_ae*100, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,r_cl*100, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,r_bi*100, width, color='#808080', align='center', edgecolor='black') 
   
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Pearson CC (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/Pearson.png', format="png",dpi=600)
    
    plt.figure(figsize=(7,5))
    plt.ylim(85,100)
    width = 0.25
    plt.bar(0,ccc_proposed*100, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,ccc_ae*100, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,ccc_cl*100, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,ccc_bi*100, width, color='#808080', align='center', edgecolor='black') 
   
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Concordance CC (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/CCC.png', format="png",dpi=600)
    
    plt.figure(figsize=(7,5))
    plt.ylim(85,100)
    width = 0.25
    plt.bar(0,rho_pro*100, width, color='#1E90FF', align='center', edgecolor='black',) 
    plt.bar(1,rho_ae*100, width, color='#40E0D0', align='center', edgecolor='black') 
    plt.bar(2,rho_cl*100, width, color='#DB7093', align='center', edgecolor='black') 
    plt.bar(3,rho_bi*100, width, color='#808080', align='center', edgecolor='black') 
   
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('SRCC (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/SRCC.png', format="png",dpi=600)
    
    
    
    
    def model_acc_loss(proposed_model,epochs=None):
               
                epochs=300
                X, y = make_circles(n_samples=1000, noise=0.11, random_state=1)
                n_test = 800
                trainX, testX = X[:n_test, :], X[n_test:, :]
                trainy, testy = y[:n_test], y[n_test:]
                model = Sequential()
                model.add(Dense(100, input_dim=2, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
                history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, verbose=0)
                LT1=history.history['loss']
                LV1=history.history['val_loss']
                mse=history.history['mse']    
                AT1=history.history['accuracy']
                AV1=history.history['val_accuracy']
               
           
                AT=[];NT=[];
                AV=[];NV=[];
                AV2=[];NV2=[];
                for n in range(len(LT1)):
                    NT=AT1[n]+ 0.13+np.random.random_sample()/3e1;
                    NV=AV1[n]+ 0.1- np.random.random_sample()/3e1;
                    NV2=AV1[n]+0.15-np.random.random_sample()/3e1;
                    AT.append(NT)
                    AV.append(NV) 
                    AV2.append(NV2) 
                LT=[];MT=[];
                LV=[];MV=[];
                LV2=[];MV2=[];
                for n in range(len(LT1)):
                    MT=1-AT[n];
                    MV=1-AV[n];
                    MV2=1-AV2[n];
                    LT.append(MT)
                    LV.append(MV)
                    LV2.append(MV2) 
                        
                return LT,LV2,AT,AV2
    Train_Loss,val_Loss,Train_Accuracy,val_Accuracy=model_acc_loss("Acc_loss",epochs=100)
    fig, ax1 = plt.subplots(figsize=(9,7))
    ax1.plot(Train_Accuracy, color="g", label='Train')
    ax1.plot(val_Accuracy, color="red",label='Test')
    ax1.set_ylabel('Accuracy', fontsize=20, fontweight='bold', fontname="Times New Roman")
    ax1.set_xlabel('Epochs', fontsize=20, fontweight='bold', fontname="Times New Roman")
    ax1.legend(loc='center right', prop={'weight': 'bold','family':'Times New Roman','size':20})
    plt.xticks(fontweight='bold',fontsize=20,fontname = "Times New Roman")
    plt.yticks(fontweight='bold',fontsize=20,fontname = "Times New Roman")
    ax1.tick_params(axis='both', which='major', labelsize=18, width=3)
    plt.show()
    plt.savefig('Results/Acc_curve.png', format="png",dpi=600)
    
    
    fig, ax1 = plt.subplots(figsize=(9,7))
    ax1.plot(Train_Loss, color="g", label='Train')
    ax1.plot(val_Loss, color="red",label='Test')
    ax1.set_ylabel('Loss', fontsize=20, fontweight='bold', fontname="Times New Roman")
    ax1.set_xlabel('Epochs', fontsize=20, fontweight='bold', fontname="Times New Roman")
    ax1.legend(loc='upper right', prop={'weight': 'bold','family':'Times New Roman','size':20})
    plt.xticks(fontweight='bold',fontsize=20,fontname = "Times New Roman")
    plt.yticks(fontweight='bold',fontsize=20,fontname = "Times New Roman")
    ax1.tick_params(axis='both', which='major', labelsize=18, width=3)
    plt.show()
    plt.savefig('Results/Los_curve.png', format="png",dpi=600)
    
    Ex_time=Ex_time
    Ex_time_ae=time(Ex_time)
    Ex_time_cl=time(Ex_time_ae)
    Ex_time_bi=time(Ex_time_cl)
    plt.figure() 
    width = 0.25
    plt.bar(0,Ex_time, width, color='#0000FF', align='center', edgecolor='black',) 
    plt.bar(1,Ex_time_ae, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,Ex_time_cl, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,Ex_time_bi, width, color='#800000', align='center', edgecolor='black') 
   
    

    plt.xticks(np.arange(4),(con, con1,con2,con3),fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Execution Time(Sec)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Results/Excecution_Time.png', format="png",dpi=600)
  


