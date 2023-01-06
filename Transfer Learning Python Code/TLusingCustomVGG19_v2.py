# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:39:05 2022

@author: Umar


modified local version of: TrasnferLearning_vgg19_modify

new in V2: extract features from images AND also SDF images to predict property
modificaiton: added loop at the end to do multiple times
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
#files = glob.glob(path + '/*.png')
#Selecting a subset of files
#files=files[0:100000] 
#imgs = [preprocess(files[i]) for i in range(len(files))]
#X_pics = np.array(imgs)


# if you don't know the names
files = glob.glob(path + '/*.png')
test_img=preprocess(files[0])
files=files[0:35]  #Selecting a subset of files IF requried
imgs=np.empty((len(files),test_img.shape[0],test_img.shape[1],test_img.shape[2]))
sdf_imgs=np.empty((len(files),test_img.shape[0],test_img.shape[1],test_img.shape[2]))
for i in range(len(files)):
    imgs[i,:,:,:]=preprocess(files[i])
    filename = files[i]
    parts = filename.split(path+'\\')
    file_sdf=path+"\\sdfs\\"+parts[-1]
    sdf_imgs[i,:,:,:]=preprocess(file_sdf)
X_pics = np.array(imgs)
X_pics_sdf = np.array(sdf_imgs)

len(X_pics)


#%% Preporceissing SDF images -NOT NEEDED ANYMORE!

# if you don't know the names
#files = glob.glob(path + '\\sdfs/*.png')
#sdf_imgs = [preprocess(files[i]) for i in range(len(files))]
#X_pics_sdf = np.array(sdf_imgs)
#%%Customize the VGG19 model for microstructure images

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


#%%Customize the VGG19 model for SDF images

from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow as tf
model_sdf=VGG19(weights='imagenet')
updated_model_sdf=Sequential()
for layer in model_sdf.layers:
    updated_model_sdf.add(layer)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>CHANGGEED BELOW!!!!!!! 
    if layer.name in 'block5_conv4': #'block5_conv4':
        break
    #if layer.name in 'block3_pool': # ONLY FOR GETTING DIRECT VALUE 
        #break
#updated_model_sdf.add(GlobalAveragePooling2D()) #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> comment out
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>invert comments
model_sdf=updated_model_sdf
model_sdf.summary()
#%% Building a model for global/averaging of select filters
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D

globall_average_model=Sequential()
globall_average_model.add(AveragePooling2D(pool_size=(5,5), strides=None, padding='same', data_format=None)) #padding='valid'
#globall_average_model.add(MaxPooling2D(pool_size=(4,4), strides=None, padding='valid', data_format=None))
#globall_average_model.add(GlobalAveragePooling2D())


globall_average_model.build([1,28,28,1]) #globall_average_model.build([1,14,14,1]) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
globall_average_model.summary()
#%% Predict
pic_features=model.predict(X_pics)
sdf_features=model_sdf.predict(X_pics_sdf) # Predict for SDF images
#%% Uncomment below!
selected_filters=[94,382,86] # selected_filters=[94,382,86,89,444]
select_sdf_features=sdf_features[:,:,:,selected_filters]
####select_sdf_features = np.expand_dims(select_sdf_features, axis=3) # not needed
select_sdf_features=globall_average_model.predict(select_sdf_features)

select_sdf_features = select_sdf_features.reshape(len(X_pics_sdf),-1)

####select_sdf_features=np.reshape(select_sdf_features,(1,-1)) # not needed

#select_sdf_features=sdf_features #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PLEASE COMMENT OUT FOR NORMAL WORK! It is only for checking out the effect of using entire SDF image and no selective kernel
#%% Do PCA
pca_original=True
if pca_original==True:
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=0.99)
    #pca_vectors = pca.fit(X_pics).transform(pic_features)
    pca_vectors = pca.fit_transform(pic_features)
    
    sum(pca.explained_variance_ratio_)
    #% Do PCA for SDF features
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=0.99)
    #pca_vectors = pca.fit(X_pics).transform(pic_features)
    pca_vectors_sdf = pca.fit_transform(select_sdf_features)
    #pca_vectors_sdf=select_sdf_features ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>REMOVE THIS
    
    sum(pca.explained_variance_ratio_)
    
#% Property extraction from file
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
df1 = pd.DataFrame(pca_vectors )
df2 = pd.DataFrame(pca_vectors_sdf)
df_master = [df1,df2] #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Change here if you want both or just one df_master = [df1, df2]

df_master = pd.concat(df_master,axis=1)

# Rename columns
a=range(0,df_master.shape[1])
new_col_names=[str(x) for x in range(df_master.shape[1])]
df_master.columns=new_col_names
#df_master=pd.DataFrame(pca_vectors) # if onlly pic data needs to be tested >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>REMOVE THIS

#%% Check correlation to get rid os more correlated terms
remove_correlated=True
if remove_correlated==True:
    
    d=df_master.transpose()
    corrs=np.corrcoef(d)
    #####to_remove=np.empty([1,len(see)])
    to_remove=[]
    for i in range(len(corrs)):
        see=corrs[:,i]
        result = np.where(see >= 0.99999) #because for some reason it shows 0.99999 instead of 1
        see[:result[0][0]+1]=0
        result = np.where(see >= 0.7)
        result=np.array(result)
        if result.size!=0:
            for j in range(result.size):
                #print(j)
                to_remove.append(result[0][j])
        #for j in range(len(result)):
    df_master.drop(df_master.columns[to_remove],axis=1,inplace=True)

#%% Normalize
from sklearn import preprocessing
x = df_master.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_master = pd.DataFrame(x_scaled)
#%
df_master['Y']=Y_property
print(df_master.head())
if save_the_file==True:
    df_master.to_csv('stepSDF_224picsNproperty.csv')
#% Building regression models

def metadata_preprocess_only_pics(df):
     X=df.drop(['Y'], axis=1)
     y=df['Y']
     return X, y

def regression_only_pics(data, model, param_grid):
     X, y = metadata_preprocess_only_pics(data)
     xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
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
#%
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
#%% ADD LOOP TO DO MULTIPLE TIMES

TEST_R_SQUARED_RF=[]
TEST_R_SQUARED_LASSO=[]
TEST_R_SQUARED_RIDGE=[]
TEST_R_SQUARED_GB=[]
TEST_R_SQUARED_GAUSSIAN=[]
for i in range(30):
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
    TEST_R_SQUARED_RF.append(R2_test_pic['rf'])
    TEST_R_SQUARED_LASSO.append(R2_test_pic['lasso'])
    TEST_R_SQUARED_RIDGE.append(R2_test_pic['ridge'])
    TEST_R_SQUARED_GB.append(R2_test_pic['gbt'])
    TEST_R_SQUARED_GAUSSIAN.append(R2_test_pic['gp'])
    print('Repition Done Times',i+1)

#%% ANALYZING RESULTS
print('TRAINING R2:{}',R2_pic)


print('TRAINING RMSE:{}',RMSE_pic)

#% Manual calculating TEST r-squared and RMSE - even though these values could be readily eextracted by R2_test_pic
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
plot_please=False
if plot_please==True:
    
    import matplotlib.pyplot as plt
    fig1=plt.figure(figsize=(15,10))
    modely='gp'
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
    
    #ax = fig1.add_subplot(1,1,1)
    #ax.get_xaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
    #ax.get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))