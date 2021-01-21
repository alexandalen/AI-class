## Brain tumor MRI analysis

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('whitegrid')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers,preprocessing
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
import warnings
warnings.filterwarnings('ignore')

#loading images

main_dir='./Brain-Tumor-Classification-DataSet-master/'
classification_dir=[('no_tumor',),('meningioma_tumor',)]
resolution=64

def load_images(root_dir_name):
    X=[]
    y=[]
    
    for label, sub_dir_names in enumerate(classification_dir):
        for sub_dir_name in sub_dir_names:
            print (f'loading {root_dir_name} {sub_dir_name}')
            sub_dir_path=os.path.join(main_dir, root_dir_name, sub_dir_name)
            
            for image_name in os.listdir(sub_dir_path):
                image_path=os.path.join(sub_dir_path, image_name)
                image= preprocessing.image.load_img(image_path,
                                                   color_mode='grayscale',
                                                   target_size=(resolution,resolution))# 64*64
                X.append(preprocessing.image.img_to_array(image))
                y.append(label)
                
    X=np.array(X)/255.0
    y=np.array(y)
    
    return X, y

#training
X_train, y_train =load_images('Training')
X_test, y_test=load_images('Testing')

X_train.shape, X_test.shape

#Visualizing Brain tumors
c=10

fig, subplots= plt.subplots(1,c)
fig.set_size_inches(20,5)
for i in range(c):
    n= np.random.randint(0, len(X_train))
    num = y_train[n]
    word= 'out' if num==0 else""
    
    subplots[i].imshow(X_train[n].reshape((resolution, resolution)),
                      cmap='gray')
    subplots[i].set_title(f'brain with{word} tumor: {num}')
    subplots[i].axis('off')
plt.show()

#Building a Convolutional neural network
input_shape=(64,64,1)
model=models.Sequential()
model.add(Conv2D(32, kernel_size=(2,2),strides=(1,1),
                 activation='linear', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64, kernel_size=(2,2),strides=(1,1),
                 activation='linear'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(128, kernel_size=(2,2),strides=(1,1),
                 activation='linear'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(256, kernel_size=(2,2),strides=(1,1),
                 activation='linear'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(512, kernel_size=(2,2),strides=(1,1),
                 activation='linear'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss= 'binary_crossentropy', metrics=['accuracy'])

train_history=model.fit(X_train,y_train, batch_size=5, epochs=10, 
          validation_data=(X_test,y_test))

plt.plot(train_history.history['loss'],'r')
plt.plot(train_history.history['val_loss'],'g')

plt.plot(train_history.history['accuracy'],'r')
plt.plot(train_history.history['val_accuracy'],'g')

#Testing the model
y_test_results= model.predict([X_test])

c=10
fig, subplots=plt.subplots(1,c)
fig.set_size_inches(25,6)
for i in range(c):
    n=np.random.randint(0, len(X_test))
    guess =str(round(y_test_results[n][0], 2)).ljust(4, '0')
    actual = y_test[n]
    
    subplot=subplots[i]
    subplot.imshow(X_test[n].reshape((resolution,resolution)), cmap='gray')
    subplot.set_title(f'predicted: {guess}, actual:{actual}')
    subplot.axis('off')
plt.show()

score = model.evaluate(X_test, y_test, verbose = 0)
score[1]

