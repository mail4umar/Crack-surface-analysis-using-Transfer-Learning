# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:34:45 2021

@author: Umar

Local version of: TrasnferLearning_vgg19_modify

To find the explainability of the images using Xiaolin's Tansfer Learning approach
"""

#%%
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
save_the_file=True # FOR SAVING FEATRE + PROPERTY CSV FILE
#%% Navigating to the respective directory
import os
cwd = os.getcwd()
path1="G:\\My Drive\\PhD work\\Northwestern University Synced All Data"
path2="\\Northwestern University\\PhD\\Code repository\\Machine Learning"
path3="\\Python\\TransferLearning\\sdf_images3\\unique"
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
files=files[0:100000] 
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
#updated_model.add(tf.keras.layers.Flatten())
#updated_model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
#updated_model.add(Dropout(0.5))
#updated_model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
model=updated_model
model.summary()

#%% Predict
pic_features=model.predict(X_pics)

#%% Do PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.99)
#pca_vectors = pca.fit(X_pics).transform(pic_features)
pca_vectors = pca.fit_transform(pic_features)

sum(pca.explained_variance_ratio_)

#%% Property extraction from file
import pandas as pd
df=pd.read_csv('properties.csv')
Y_property=np.zeros((len(files),1))
for i in range(len(files)):
  filename = files[i]
  parts = filename.split(path+'\\')
  nameonly = parts[len(parts)-1].split('.')[0]
  number=int(nameonly)
  q=df.loc[df['serial'] == number]['y']
  Y_property[i]=q.values

#%% Makes master dataframe 
df_master = pd.DataFrame(pca_vectors)
df_master['Y']=Y_property
print(df_master.head())
if save_the_file==True:
    df_master.to_csv('stepSDF_224picsNproperty.csv')
#%% Building regression models

def metadata_preprocess_only_pics(df):
     X=df.drop(['Y'], axis=1)
     y=df['Y']
     return X, y

def regression_only_pics(data, model, param_grid):
     X, y = metadata_preprocess_only_pics(data)
     xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
     reg = GridSearchCV(model, param_grid, cv=10)
     reg.fit(xtrain, ytrain)
     ypred = reg.predict(xtest)
     r2 = reg.score(xtrain, ytrain) # r2 = reg.score(xtrain, ytrain) for training r-squared
     r2_test = reg.score(xtest, ytest)
     rmse = np.sqrt(mean_squared_error(ytest, ypred))
     mae = mean_absolute_error(ytest,ypred)

     return reg, ypred, ytest, r2, rmse, mae, xtest, r2_test

# def regression_only_pics_for_GP(data, model):
#      X, y = metadata_preprocess_only_pics(data)
#      xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
#      model.fit(xtrain, ytrain)
#      ypred = reg.predict(xtest)
#      r2 = reg.score(xtrain, ytrain) # r2 = reg.score(xtrain, ytrain) for training r-squared
#      r2_test = reg.score(xtest, ytest)
#      rmse = np.sqrt(mean_squared_error(ytest, ypred))
#      mae = mean_absolute_error(ytest,ypred)

#      return reg, ypred, ytest, r2, rmse, mae, xtest, r2_test
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import matplotlib.pyplot as plt
#from linear_model import Ridge
#from sklearn.ensemble import Ridge
#from sklearn.ensemble import Lasso
models = {'gbt': GradientBoostingRegressor(),
                 'rf': RandomForestRegressor(),
                 'ridge': linear_model.Ridge(),
                 'lasso': linear_model.Lasso(),
                 'gp': GaussianProcessRegressor()
                 }

param_grids={'gbt': [{'learning_rate': [0.05,0.1,0.2], 'n_estimators': [50,100,200]}],
                 'rf': [{'n_estimators': [10,200,300], 'max_features': ['auto','sqrt','log2']}], #n_estimators': [10,200,300]
                 'ridge': [{'alpha': [1e-2, 2e-2],'max_iter': [100000]}],
                 'lasso': [{'alpha': [1e-2, 1e-3],'max_iter': [100000]}],
                'gp': [{'alpha': [1e-3,1e-10]}]
                 }

keys = list(models.keys( ))
ypreds_pic, R2_pic, RMSE_pic, MAE_pic,R2_test_pic,ytests_pic = {}, {}, {}, {},{},{}
for key in keys:
     model = models[key]
     param_grid = param_grids[key]
     model, ypred, ytest, r2, rmse, mae, xtest,r2_test = regression_only_pics(df_master, model, param_grid)
     models[key] = model
     ypreds_pic[key] = ypred
     ytests_pic[key]=ytest
     R2_pic[key] = r2
     R2_test_pic[key]=r2_test
     RMSE_pic[key] = rmse
     MAE_pic[key] = mae
     
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

print('')
print ('RMSE')
print('RANDOM FOREST : ',rmse_rf)
print('LASSO : ',rmse_l)
print('RIDGE Regression : ',rmse_r)
print('Gradient Boosting : ',rmse_g)
print('Gaussian Process : ',rmse_gp)
RMSE_pic

#%% Plotting
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