# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:00:48 2022

@author: Umar

modified local version of: TrasnferLearning_vgg19_modify
new in v2_3: added blind test, more autoamtion

new in V2: extract features from images AND also SDF images to predict property
modificaiton: added loop at the end to do multiple times

new in V2_2: added linear regression
"""

#%%
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
save_the_file=True # FOR SAVING FEATRE + PROPERTY CSV FILE
scale_property=True # FOR SCALING VERY HIGH VALUES LIKE E
desired_block_images='block4_pool'
desired_block_sdf_images='block5_pool'
pca_original=True # if we need to do PCA
correlation_CUTOFF=0.5

remove_correlated=True #if we want to remove correlated terms
PCA_components_images=11    
PCA_components_sdf_images=8
Blind_test=True
property_file='da_dn.csv'

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
#files = glob.glob(path + '/*.png')
#Selecting a subset of files
#files=files[0:100000] 
#imgs = [preprocess(files[i]) for i in range(len(files))]
#X_pics = np.array(imgs)


# if you don't know the names
files = glob.glob(path + '/*.png')
test_img=preprocess(files[0])
files=files[0:1000]  #Selecting a subset of files IF requried
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
    if layer.name in desired_block_images: #block3_conv4
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
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>CHANGGEED BELOW!!!!!!! Invert comments tif you want only SDF whole
#    if layer.name in 'block5_conv4': #'block5_conv4':
#        break
    if layer.name in desired_block_sdf_images: # ONLY FOR GETTING DIRECT VALUE 
        break
updated_model_sdf.add(GlobalAveragePooling2D()) #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> comment out
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>invert comments
model_sdf=updated_model_sdf
model_sdf.summary()
#%% Building a model for global/averaging of select filters
#from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D

#globall_average_model=Sequential()
#globall_average_model.add(AveragePooling2D(pool_size=(5,5), strides=None, padding='same', data_format=None)) #padding='valid'
#globall_average_model.add(MaxPooling2D(pool_size=(4,4), strides=None, padding='valid', data_format=None))
#globall_average_model.add(GlobalAveragePooling2D())


#globall_average_model.build([1,28,28,1]) #globall_average_model.build([1,14,14,1]) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#globall_average_model.summary()
#%% Predict
pic_features=model.predict(X_pics)
sdf_features=model_sdf.predict(X_pics_sdf) # Predict for SDF images
#%% Uncomment below! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Invert comments tif you want only SDF whole
#selected_filters=[94,382,86] # selected_filters=[94,382,86,89,444]
#select_sdf_features=sdf_features[:,:,:,selected_filters]
####select_sdf_features = np.expand_dims(select_sdf_features, axis=3) # not needed
#select_sdf_features=globall_average_model.predict(select_sdf_features)

#select_sdf_features = select_sdf_features.reshape(len(X_pics_sdf),-1)

####select_sdf_features=np.reshape(select_sdf_features,(1,-1)) # not needed

select_sdf_features=sdf_features #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>PLEASE COMMENT OUT FOR NORMAL WORK! It is only for checking out the effect of using entire SDF image and no selective kernel
#%% Do PCA

if pca_original==True:
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=PCA_components_images)
    #pca_vectors = pca.fit(X_pics).transform(pic_features)
    pca_vectors = pca.fit_transform(pic_features)
    print('PCAs from pic microstructure images are:',pca_vectors.shape[1])
    print("image variance explained=",sum(pca.explained_variance_ratio_))
    #% Do PCA for SDF features
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=PCA_components_sdf_images)
    #pca_vectors = pca.fit(X_pics).transform(pic_features)
    pca_vectors_sdf = pca.fit_transform(select_sdf_features)
    print('PCAs from SDF images are:',pca_vectors_sdf.shape[1])
    #pca_vectors_sdf=select_sdf_features ####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>REMOVE THIS
    
    print("sdf image variance explaiend=",sum(pca.explained_variance_ratio_))
    
#% Property extraction from file
import pandas as pd
df=pd.read_csv(property_file)
Y_property=np.zeros((len(files),1))
for i in range(len(files)):
  filename = files[i]
  parts = filename.split(path+'\\')
  nameonly = parts[len(parts)-1].split('.')[0]
  nameonly = nameonly.split('_')[0]
  number=int(nameonly)
  q=df.loc[df['Serial'] == number]['y']
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

    

if remove_correlated==True:
    
    d=df_master.transpose()
    corrs=np.corrcoef(d)
    if len(corrs)!=1: # If number of variables is only 1 then no need for correlation analysis
        #####to_remove=np.empty([1,len(see)])
        to_remove=[]
        for i in range(len(corrs)):
            see=corrs[:,i]
            result = np.where(see >= 0.99999) #because for some reason it shows 0.99999 instead of 1
            see[:result[0][0]+1]=0
            result = np.where(see >= correlation_CUTOFF)
            result=np.array(result)
            if result.size!=0:
                for j in range(result.size):
                    #print(j)
                    to_remove.append(result[0][j])
            #for j in range(len(result)):
        df_master.drop(df_master.columns[to_remove],axis=1,inplace=True)
print("After rremoving correlated inputs are reduced to:",df_master.shape[1])
df_master.head()

#%% Normalize
from sklearn import preprocessing
x = df_master.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_master = pd.DataFrame(x_scaled)
#%
if scale_property==True:
    Y_property = min_max_scaler.fit_transform(Y_property)
df_master['Y']=Y_property
print(df_master.head())
if save_the_file==True:
    df_master.to_csv('stepSDF_224picsNproperty.csv')
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
for i in range(30):
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