# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:48:00 2022

@author: mail4



Local version of: TrasnferLearning_vgg19_modify

To find the explainability of the images using Xiaolin's Tansfer Learning approach

new in v1_4:
    Trying to extract features from multiple blocks and then applying PCA
    CONCERN: not sure if it takes h2 nicely

new in v1_3:
    modification: tryiing to split test and training data so that some samples remain hidden
    thisis because there are multiple images of same point (this can be changed around line 168 - 172)
    This has been called 'Blind_test' = The data is truly hidden
    modification: added dropping image feature to get rid of unwated datasets
new in v1_2:
modificaiton: added loop at the end to do multiple times
modification: added linear regression
modification: changed property file name
"""

#%%
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
save_the_file=True # FOR SAVING FEATRE + PROPERTY CSV FILE
scale_property=True # FOR SCALING VERY HIGH VALUES LIKE E
Extra_Input=False # If you like to add extra inputs along with iamge
Use_Both_Inputs=False # If you like to use both images and second property like h2,temp etc
Use_Only_Images=True # If you only like to use iamges
Use_Only_Second_Input=False # if you just want to use the second property
Blind_test=True # TO ensure the data is truly hidden when splitting test and training

Drop_Images=[]
if len(Drop_Images)>=1:
    Drop_Images_Please=True
else:
    Drop_Images_Please=False
#%% Navigating to the respective directory
import os
cwd = os.getcwd()
path1="D:\\My Drive\\PhD work\\Northwestern University Synced All Data"
path2="\\Northwestern University\\PhD\\Code repository\\Machine Learning"
path3="\\Python\\TransferLearning\\ford\\HT_Trans_PNG"
path=path1+path2+path3
try:
    os.chdir(path)
    print("Current working directory: {0}".format(os.getcwd()))
except FileNotFoundError:
    print("Directory: {0} does not exist".format(path))
except NotADirectoryError:
    print("{0} is not a directory".format(path))
except PermissionError:
    print("You do not have permissions to change to {0}".format(path))
    
#%% Preporceissing images

DECREASED_SIZE=224 # 224 
def preprocess(file):
     try:
         img = image.load_img(file, target_size=(DECREASED_SIZE, DECREASED_SIZE))
         feature = image.img_to_array(img)
         feature = np.expand_dims(feature, axis=0)
         feature = preprocess_input(feature)
         return feature[0]
     except:
         print('Error:', file)
         
import glob
# if you don't know the names
files = glob.glob(path + '/*.png')
#Selecting a subset of files
#%% Remove/Drop unwanted files
files=files[0:100000] 
import pandas as pd
if Drop_Images_Please:
    dropping_index=[]
    for i in range(len(files)):
      filename = files[i]
      parts = filename.split(path+'\\')
      nameonly = parts[len(parts)-1].split('.')[0]
      nameonly = nameonly.split('_')[0]
      if nameonly in Drop_Images:
          #print(nameonly)
          #print(i)
          dropping_index.append(i)
    dropping_index=list(set(dropping_index))
    files=pd.DataFrame(files)
    files=files.drop(dropping_index)
    files=list(files[0])
#%% Read Images
imgs = [preprocess(files[i]) for i in range(len(files))]
X_pics = np.array(imgs)

len(imgs)

#%%Customize the VGG19 model

from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow as tf
model=VGG19(weights='imagenet')
updated_model=Sequential()
for layer in model.layers:
    updated_model.add(layer)
    if layer.name in 'block3_pool': #block3_conv4
        break


updated_model.add(GlobalAveragePooling2D())
model=updated_model
model.summary()

#%%Customize the VGG19 model # 2

from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow as tf
model2=VGG19(weights='imagenet')
updated_model2=Sequential()
for layer in model2.layers:
    updated_model2.add(layer)
    if layer.name in 'block4_pool': #block3_conv4
        break


updated_model2.add(GlobalAveragePooling2D())
model2=updated_model2
model2.summary()

#%%Customize the VGG19 model model # 3

from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow as tf
model3=VGG19(weights='imagenet')
updated_model3=Sequential()
for layer in model3.layers:
    updated_model3.add(layer)
    if layer.name in 'block5_pool': #block3_conv4
        break


updated_model3.add(GlobalAveragePooling2D())
model3=updated_model3
model3.summary()
#%% Predict
pic_features=model.predict(X_pics)
pic_features2=model2.predict(X_pics)
pic_features3=model3.predict(X_pics)

pic_features=np.append(pic_features,pic_features2,axis=1)
pic_features=np.append(pic_features,pic_features3,axis=1)
#%% Do PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)


#pca_vectors = pca.fit(X_pics).transform(pic_features)
pca_vectors = pca.fit_transform(pic_features)

sum(pca.explained_variance_ratio_)

#%% Property extraction from file
import pandas as pd
df=pd.read_csv('da_dn.csv')  # Change da_dn here to get another property
Y_property=np.zeros((len(files),1))
for i in range(len(files)):
  filename = files[i]
  parts = filename.split(path+'\\')
  nameonly = parts[len(parts)-1].split('.')[0]
  nameonly = nameonly.split('_')[0]
  number=int(nameonly)
  q=df.loc[df['Serial'] == number]['y']
  Y_property[i]=q.values[0]
  
if Extra_Input:
    df=pd.read_csv('h2.csv')  # Change da_dn here to get another property
    X_2=np.zeros((len(files),1))
    for i in range(len(files)):
      filename = files[i]
      parts = filename.split(path+'\\')
      nameonly = parts[len(parts)-1].split('.')[0]
      nameonly = nameonly.split('_')[0]
      number=nameonly
#      number=int(nameonly)
      q=df.loc[df['Serial'] == number]['h2']
      X_2[i]=q.values[0]

#%% Makes master dataframe 

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
if Extra_Input:
    X_2=min_max_scaler.fit_transform(X_2)
if Use_Both_Inputs:
    df_master = pd.DataFrame(pca_vectors)
    x = df_master.values #returns a numpy array
    x_scaled = min_max_scaler.fit_transform(x)
    df_master = pd.DataFrame(x_scaled)
    df_master['x2']=X_2
elif Use_Only_Images:
    df_master = pd.DataFrame(pca_vectors)
    x = df_master.values #returns a numpy array
    x_scaled = min_max_scaler.fit_transform(x)
    df_master = pd.DataFrame(x_scaled)
elif Use_Only_Second_Input:            
    df_master = pd.DataFrame(X_2)

if scale_property:
    Y_property = min_max_scaler.fit_transform(Y_property)

##df_master = pd.DataFrame(X_2)

df_master['Y']=Y_property

print(df_master.head())
if save_the_file==True:
    df_master.to_csv('features_hieght_34picsNproperty.csv')
#%% Building regression models

def metadata_preprocess_only_pics(df):
     X=df.drop(['Y'], axis=1)
     y=df['Y']
     return X, y

def train_test_splitter(df):
    a=pd.unique(df.Y)
    trainer,tester=train_test_split(a,test_size=0.2, shuffle=True)
    return trainer,tester

def regression_only_pics(data, model, param_grid,blind): # added option to blind test or not
     X, y = metadata_preprocess_only_pics(data)
     # For regular datasets without repetition:
     xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
     # For repition datasets
     trainer,tester=train_test_splitter(data)
     if blind:
         xtrain=X[data['Y'].isin(trainer)]
         xtest=X[data['Y'].isin(tester)]
         ytest=y[data['Y'].isin(tester)]
         ytrain=y[data['Y'].isin(trainer)]
         
     reg = GridSearchCV(model, param_grid, cv=10)
     reg.fit(xtrain, ytrain)
     ypred = reg.predict(xtest)
     r2 = reg.score(xtrain, ytrain) # r2 = reg.score(xtrain, ytrain) for training r-squared
     r2_test = reg.score(xtest, ytest)
     rmse = np.sqrt(mean_squared_error(ytest, ypred))
     mae = mean_absolute_error(ytest,ypred)
     return reg, ypred, ytest, r2, rmse, mae, xtest, r2_test
#%% initialize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#from linear_model import Ridge
#from sklearn.ensemble import Ridge
#from sklearn.ensemble import Lasso
#%% ADD LOOP TO DO MULTIPLE TIMES

TEST_R_SQUARED_RF=[]
TEST_R_SQUARED_LASSO=[]
TEST_R_SQUARED_RIDGE=[]
TEST_R_SQUARED_GB=[]
TEST_R_SQUARED_GAUSSIAN=[]
TEST_R_SQUARED_LINEAR=[]
for i in range(35):
    models = {'gbt':GradientBoostingRegressor(),
                     'rf':RandomForestRegressor(),
                     'ridge':linear_model.Ridge(),
                     'lasso':linear_model.Lasso(),
                     'gp':GaussianProcessRegressor(),
                     'lr':LinearRegression()
                     }
    param_grids={'gbt':[{'learning_rate': [0.05,0.1,0.2], 'n_estimators': [50,100,200]}],
                     'rf':[{'n_estimators': [10,200,300], 'max_features': ['auto','sqrt','log2']}], #n_estimators': [10,200,300]
                     'ridge':[{'alpha': [1e-2, 2e-2,0.1],'max_iter': [100000]}],
                     'lasso':[{'alpha': [1e-2, 1e-3],'max_iter': [100000]}],
                    'gp':[{'alpha': [1e-3,1e-10]}],
                    'lr':[{'normalize':['False','True']}], #'lr':[{'normalize':['True']}]
                     }
    keys = list(models.keys( ))
    ypreds_pic, R2_pic, RMSE_pic, MAE_pic,R2_test_pic,ytests_pic = {}, {}, {}, {},{},{}
    for key in keys:
         model = models[key]
         param_grid = param_grids[key]
         model, ypred, ytest, r2, rmse, mae, xtest,r2_test = regression_only_pics(df_master, model, param_grid,Blind_test)
         models[key] = model
         ypreds_pic[key] = ypred
         ytests_pic[key]=ytest
         R2_pic[key] = r2
         R2_test_pic[key]=r2_test
         RMSE_pic[key] = rmse
         MAE_pic[key] = mae
    TEST_R_SQUARED_RF.append(R2_test_pic['rf'])
    TEST_R_SQUARED_LASSO.append(R2_test_pic['lasso'])
    TEST_R_SQUARED_RIDGE.append(R2_test_pic['ridge'])
    TEST_R_SQUARED_GB.append(R2_test_pic['gbt'])
    TEST_R_SQUARED_GAUSSIAN.append(R2_test_pic['gp'])
    TEST_R_SQUARED_LINEAR.append(R2_test_pic['lr'])
    print('Repition Done Times',i+1)
import statistics
print('Average R-squared values:')
print('RF',statistics.mean(TEST_R_SQUARED_RF))
print('Lasso',statistics.mean(TEST_R_SQUARED_LASSO))
print('Ridge',statistics.mean(TEST_R_SQUARED_RIDGE))
print('GB',statistics.mean(TEST_R_SQUARED_GB))
print('Gaussian',statistics.mean(TEST_R_SQUARED_GAUSSIAN))
print('Linear',statistics.mean(TEST_R_SQUARED_LINEAR))

import statistics
print('Standard Deviations:')
print('RF',statistics.stdev(TEST_R_SQUARED_RF))
print('Lasso',statistics.stdev(TEST_R_SQUARED_LASSO))
print('Ridge',statistics.stdev(TEST_R_SQUARED_RIDGE))
print('GB',statistics.stdev(TEST_R_SQUARED_GB))
print('Gaussian',statistics.stdev(TEST_R_SQUARED_GAUSSIAN))
print('Linear',statistics.stdev(TEST_R_SQUARED_LINEAR))
#%% ANALYZING RESULTS
print('TRAINING R2:{}',R2_pic)


print('TRAINING RMSE:{}',RMSE_pic)

#%% Manual calculating TEST r-squared and RMSE - even though these values could be readily eextracted by R2_test_pic
import sklearn
import math
print('TEST SET RESULTS!')
yyy=ytests_pic['rf']
f=ypreds_pic['rf']
ss_tot=sum((yyy-yyy.mean())**2)
ss_res=sum((yyy-f)**2)
r_squared=1-ss_res/ss_tot
mse_rf = sklearn.metrics.mean_squared_error(yyy, f)
rmse_rf = math.sqrt(mse_rf)

print('R-squared values')
print('RANDOM FOREST : ',r_squared)

yyy=ytests_pic['lasso']
f=ypreds_pic['lasso']
ss_tot=sum((yyy-yyy.mean())**2)
ss_res=sum((yyy-f)**2)
r_squared=1-ss_res/ss_tot
mse_l = sklearn.metrics.mean_squared_error(yyy, f)
rmse_l = math.sqrt(mse_l)
print('LASSO - : ',r_squared)

yyy=ytests_pic['ridge']
f=ypreds_pic['ridge']
ss_tot=sum((yyy-yyy.mean())**2)
ss_res=sum((yyy-f)**2)
r_squared=1-ss_res/ss_tot
mse_r = sklearn.metrics.mean_squared_error(yyy, f)
rmse_r = math.sqrt(mse_r)
print('RIDGE Regression - : ',r_squared)

yyy=ytests_pic['gbt']
f=ypreds_pic['gbt']
ss_tot=sum((yyy-yyy.mean())**2)
ss_res=sum((yyy-f)**2)
r_squared=1-ss_res/ss_tot
mse_g = sklearn.metrics.mean_squared_error(yyy, f)
rmse_g = math.sqrt(mse_g)
print('Gradient Boosting : ',r_squared)

yyy=ytests_pic['gp']
f=ypreds_pic['gp']
ss_tot=sum((yyy-yyy.mean())**2)
ss_res=sum((yyy-f)**2)
r_squared=1-ss_res/ss_tot
mse_gp = sklearn.metrics.mean_squared_error(yyy, f)
rmse_gp = math.sqrt(mse_gp)
print('Gaussian Process : ',r_squared)
print('Linear Regression : ',R2_test_pic['lr'])
print('')
print ('RMSE')
print('RANDOM FOREST : ',rmse_rf)
print('LASSO : ',rmse_l)
print('RIDGE Regression : ',rmse_r)
print('Gradient Boosting : ',rmse_g)
print('Gaussian Process : ',rmse_gp)
RMSE_pic

#%% Plotting
please_plot=False
if please_plot==True:
        
    import matplotlib.pyplot as plt
    fig1=plt.figure(figsize=(15,10))
    modely='rf'
    ally=ytests_pic[modely]
    maxy=ally.max()
    miny=ally.min()
    plt.scatter(ytests_pic[modely],ypreds_pic[modely], label = 'Data')
    plt.plot([miny,maxy],[miny,maxy],'r', label = 'Perfect Match', linewidth=2)
    
    plt.xlabel('Actual Value',fontsize=20)
    plt.ylabel('Predicted Value',fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ax = fig1.add_subplot(1,1,1)
    #ax.get_xaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
    #ax.get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))