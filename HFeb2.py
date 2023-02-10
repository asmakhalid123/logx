#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install ipython ipykernel
get_ipython().run_line_magic('pip', 'install pillow')
get_ipython().run_line_magic('pip', 'install PIL')
get_ipython().run_line_magic('pip', 'install tensorflow')
import pandas as pd
import os
os.getcwd()
from PIL import Image
get_ipython().run_line_magic('pip', 'install -U imbalanced-learn')
get_ipython().run_line_magic('pip', 'install seaborn')


# In[2]:


conda install dask


# In[27]:


import csv
import re
import pandas as pd
import os
import cv2
import time
import PIL
import PIL.Image
import sys
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import re
import warnings
import logging
import os
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras import datasets, layers, models # I added this one
from tensorflow.keras.models import Sequential # I just added this one 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, GlobalMaxPooling2D, Flatten, MaxPooling2D
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
# from keras import optimizers
from sklearn.decomposition import PCA
import seaborn as sns
# import keras
# from keras import backend as K
# from keras.layers import Dense, Dropout, BatchNormalization, Conv2D
# from keras.optimizers import RMSprop, Adam, SGD
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # I added this 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
# from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')
print('Working directory:', os.getcwd())


# In[ ]:


# from keras import Sequential 
# from keras.layers import Dense, Dropout, GlobalMaxPooling2D, Flatten, MaxPooling2D
# from keras import optimizers
# from keras import backend as K
# from keras.layers import Dense, Dropout, BatchNormalization, Conv2D
# from keras.optimizers import RMSprop, Adam, SGD


# In[28]:


path="E:\\Internship\\Project\\" 
dirs=os.listdir(path)
dirs.sort()


# In[81]:


def create_dataset():
    imgset=[]
    for item in dirs:
        if os.path.isfile(path+item):
            im= PIL.Image.open(path + item).convert("RGB")
            im=np.array(im)
#             im.resize([128,128])
            im = cv2.resize(im, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            
            
            
            imgset.append(im)
#             print(path+item)
            # print("found")
#         else:
            # print(path+item)
    np.save("new_box.npy", imgset)


# In[82]:


imgset=create_dataset()


# In[83]:


img_array=np.load("new_box.npy")
#"E:\Internship\\","new_box.npy"))#c:\Users\\HUMA\\Downloads\\


# In[84]:


from matplotlib import pyplot as plt
plt.imshow(img_array[1], cmap='gray')
plt.show()


# In[85]:


print(img_array.shape)


# In[86]:


sz=len(img_array)
print(sz)


# In[87]:


# Preprocess images using gaussian blur filtering

gaussian = img_array.copy()/255
for i in range(sz):
  #gaussian[i] = img_array[i]
  gaussian[i] = cv2.blur(gaussian[i],(1,1))

  #gaussian[i] = cv2. #cv2.GaussianBlur(gaussian[i],(1,1),0)
  #gaussian[i] = cv2.Laplacian(gaussian[i],cv2.CV_64F)
  #gaussian[i] = cv2.Sobel(gaussian[i],cv2.CV_64F,1,0,ksize=1)  # y
  #gaussian[i] = cv2.Sobel(gaussian[i],cv2.CV_64F,0,1,ksize=1)  # x


for i in range(3):
  plt.figure(figsize=(12,12))
  plt.subplot(2, 3, i + 1)
  plt.imshow(gaussian[i])
     
     


# In[88]:


colname=['Label']
df=pd.read_csv('new_box.csv', names=colname, header=None)
df.shape


# In[89]:


normal = gaussian.copy()
X = normal
y = df 


# In[90]:


plt.imshow(X[1])


# In[91]:


print(X.shape)


# In[92]:


X = np.transpose(X, (0, 2, 1, 3))


# In[79]:


plt.imshow(X[1])


# In[93]:


print (X.shape)


# In[74]:


# X = np.resize(X, (499,128,128, 1))
# print (X.shape)


# In[159]:


plt.imshow(X[0])


# y.shape
# y

# # Create a dictionary for our softmax lookup

# In[95]:


box_dict = {0:'0B',1:'30',2:'B0', 3:'B73', 4:'B75', 5:'JC', 6:'P4', 7:'U3', 8:'U030', 9:'V406'}


# In[96]:


for i in range(len(box_dict)):
    y= y.replace(box_dict[i], i)
print(len(box_dict))
     


# In[97]:



y.head(220)


# In[145]:


df.tail(200)


# In[98]:


df.head()


# In[99]:


df.info


# Transform y into categorical values in relation to our softmax target output

# In[100]:


from tensorflow.keras.utils import to_categorical as tc
smTarget = len(box_dict)

y = tc(y, num_classes=smTarget)
seed = 7
check = 498


# # Split our test and training set using 0.3

# In[101]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


# In[103]:


print(X_train.shape)
print(X_test.shape)
print(X.shape)


# In[105]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
     


# In[107]:


plt.imshow(X_train[1])


# In[45]:


# from skimage import io
# img=X_train[1].reshape(128,128)
# io.imshow("1",img)


# # Initialise optimizer

# In[160]:


from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Sequential

adam = Adam(learning_rate=0.001)


# Build Model

# In[161]:


inp_sh = (128, 128, 3)
ks = (5,5)
model = Sequential()
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=ks, padding='same', input_shape=inp_sh, activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Conv2D(128, ks, padding='same', activation="relu")) # 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3)))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Conv2D(128, ks, padding='same', activation="relu")) # 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=ks, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=ks, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=ks, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=ks, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.GlobalMaxPooling2D())
model.add(tf.keras.layers.Flatten()) # Flatten the image so we can pass it to our fully connected layer
model.add(tf.keras.layers.Dense(512, activation='relu')) 
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(smTarget, activation="softmax"))


# # Compile Model:

# In[162]:


model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
model.build( input_shape=X_train.shape)
model.summary()


# In[163]:


X_train = X_train.astype('float32') # Conversion to float type from integer type.
X_test = X_test.astype('float32')
#X_train /= 255.0 # Division by 255
#X_test /= 255.0
print(X_train.shape)
print(X_test.shape)
print(X.shape)


# In[164]:


import tensorflow as tf
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5000)


# In[165]:


model_checkpoint =  tf.keras.callbacks.ModelCheckpoint('cifar_cnn_checkpoint_{epoch:02d}_loss{loss:.4f}.h5',
                                                           monitor='val_accuracy',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='auto',
                                                           save_freq=1)
                                                           


# # Create image generator and fit it to train the training data

# In[166]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 180, zoom_range = 0.2, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True, vertical_flip = True)
generator.fit(X_train)
print(X_train.shape)
print(X_test.shape)


# In[167]:


lrr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                        patience=50, 
                        verbose=1, 
                        factor=0.3, 
                        min_lr=0.0001)
filepath="E://Internship/weights.best.best_{epoch:02d}-{accuracy:.2f}.h5"
callbacks_list = [lrr, model_checkpoint]#, early_stopping]


# # Train the model on test data

# In[168]:


print(X_train.shape)


# In[169]:


print(X_test.shape)


# In[173]:


history = model.fit( x=X_train, y=y_train, epochs=10, batch_size=10, steps_per_epoch=49,  validation_data=(X_test, y_test), callbacks=callbacks_list )


# In[182]:


model.save("my_model2")


# In[137]:


print(X_test.shape)


# # View the shape of our training set and our test set

# In[174]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
y_pred = model.predict(X_test)
y_class = np.argmax(y_pred, axis = 1) 
y_check = np.argmax(y_test, axis = 1) 
cmatrix = confusion_matrix(y_check, y_class)
cmatrix


# # Visualize a heatmap of our confusion matrix

# In[175]:


import seaborn as sns
sns.heatmap(cmatrix, xticklabels=box_dict)


# Visualize the prediction
# 

# In[176]:


def predict(check):
  index = 0
  highest = 0
  for i in range(len(box_dict)):
    v = y_pred[check][i]
    if( v > highest):
      highest = v
      index = i
  print("index:", i)
  print("highest:", highest)
  print("check:", check)
  print("predicted:", box_dict[index], " @ index ", index, " in ", y_pred[index])
  print("actual:", box_dict[index], " @ index ", index, " in ", y_test[index])
  print("reference:", box_dict)


# In[177]:


predict(2)


# In[178]:


predict(3)


# In[179]:


predict(30)


# In[180]:


predict(28)


# In[ ]:





# In[ ]:





# In[ ]:



    


# Map our labels to softmax numbers
# 

# In[ ]:





# In[ ]:




