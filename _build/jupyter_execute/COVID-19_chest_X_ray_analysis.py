## COVID-19 chest X ray practice 

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import * #(layers, models, optimizers,preprocessing)
from tensorflow.keras.layers import * #(Conv2D, Dense, MaxPooling2D, Flatten, Dropout)
from tensorflow.keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

main_dir='./chest_xray/chest_xray/'
classification_dir=[('NORMAL'),('PNEUMONIA')]
resolution=128
def load_images(root_dir_name):
    X=[]
    y=[]
    
    for label, sub_dir_name in enumerate(classification_dir):
        print (f'loading {root_dir_name} {sub_dir_name}')
        sub_dir_path=os.path.join(main_dir, root_dir_name, sub_dir_name)
            
        for image_name in os.listdir(sub_dir_path):
            if label==1 and 'virus' in image_name: continue
            image_path= os.path.join(sub_dir_path, image_name)
            image= preprocessing.image.load_img(image_path,
                                                color_mode='grayscale',
                                                target_size=(resolution,resolution))#128*128
            X.append(preprocessing.image.img_to_array(image))
            y.append(label)
    X=np.array(X)/255.0
    y=np.array(y)
    
    return X,y

# Training
X_train, y_train =load_images('train')
X_test, y_test = load_images('test')

# Visualizing Brain tumors
c=8

fig, subplots = plt.subplots(1,c)
fig.set_size_inches(20,5)
for i in range(c):
    n= np.random.randint(0, len(X_train))
    num = y_train[n]
    word= 'healthy' if num==0 else"pneumonia"
    
    subplots[i].imshow(X_train[n].reshape((resolution,resolution)),
                      cmap='gray')
    subplots[i].set_title(f'{word}: {num}')
    subplots[i].axis('off')
plt.show()


#Building a Convolutional neural network
input_shape=(resolution,resolution,1)
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

train_history=model.fit(X_train,y_train, batch_size=8, epochs=10, 
          validation_data=(X_test,y_test))

plt.plot(train_history.history['loss'],'r')
plt.plot(train_history.history['val_loss'],'g')

plt.plot(train_history.history['accuracy'],'r')
plt.plot(train_history.history['val_accuracy'],'g')

os.listdir('./chest_xray/train')

#Data Visualization

train_dir='./chest_xray/train'
test_dir='./chest_xray/test'
val_dir='./chest_xray/val'

print('Train set:\n--------------------------------')
num_pneumonia=len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
num_normal=len(os.listdir(os.path.join(train_dir,'NORMAL')))
print(f'PNEUMONIA={num_pneumonia}')
print(f'NORMAL={num_normal}')

print('Test set:\n---------------------------------')
print(f"PNEUMONIA={len(os.listdir(os.path.join(test_dir, 'PNEUMONIA')))}")
print(f'PNEUMONIA={len(os.listdir(os.path.join(test_dir, "NORMAL")))}')

print('Validation set:\n---------------------------')
print(f'PNEUMONIA={len(os.listdir(os.path.join(val_dir, "PNEUMONIA")))}')
print(f'PNEUMONIA={len(os.listdir(os.path.join(val_dir, "NORMAL")))}')

pneumonia= os.listdir('./chest_xray/train/PNEUMONIA')
pneumonia_dir='./chest_xray/train/PNEUMONIA'

plt.figure(figsize=(20,10))

for i in range(9):
    plt.subplot(3,3,i+1)
    img=plt.imread(os.path.join(pneumonia_dir,pneumonia[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.tight_layout()

normal=os.listdir('./chest_xray/train/NORMAL')
normal_dir='./chest_xray/train/NORMAL'

plt.figure(figsize=(20,10))

for i in range(9):
    plt.subplot(3,3,i+1)
    img= plt.imread(os.path.join(normal_dir, normal[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
plt.tight_layout()

normal_img=os.listdir('./chest_xray/train/NORMAL')[1]
normal_dir='./chest_xray/train/NORMAL'
sample_img=plt.imread(os.path.join(normal_dir,normal_img))
plt.imshow(sample_img, cmap='gray')
plt.colorbar()
plt.title('Raw Chest X-RAY image')

print(f"The dimensions of the image are {sample_img.shape[1]} pixels width and {sample_img.shape[1]} pixels height,one single color channel.")
print(f"The maximum pixel value is {sample_img.max():.4f} and the minimum is {sample_img.min():.4f}")
print(f"The mean value of the pixels is {sample_img.mean():.4f} and he standard deviation is {sample_img.std():.4f}")


# investigate pixel value distribution
sns.distplot(sample_img.ravel(),
             label=f"Pixel Mean {np.mean(sample_img):.4f} and STD {np.std(sample_img):.4f}",
             kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixel in Image')

#img preprocessing
'''
Before training, we'll first modify your images to be better suited for 
training a convolutional neural network.
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_generator= ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True)

#Build a separate generator for valid and test sets
'''
flow_from_directory:以文件夾路徑為參數，生成通過數據提升/歸一化後的數據，在一個無限循環中無限產生batch數據
flow_from_dataframe:輸入dataframe和目錄的路徑，並生成批量的增強/標準化的數據。
'''
train= image_generator.flow_from_directory(train_dir,
                                           batch_size=8,
                                           shuffle=True,
                                           class_mode='binary',
                                           target_size=(180,180))
validation=image_generator.flow_from_directory(val_dir,
                                               batch_size=1,
                                               shuffle=False,
                                               class_mode='binary',
                                               target_size=(180,180))
test=image_generator.flow_from_directory(test_dir,
                                         batch_size=1,
                                         shuffle=False,
                                         class_mode='binary',
                                        target_size=(180,180))


sns.set_style('white')
generated_image, label= train.__getitem__(2)
plt.imshow(generated_image[2], cmap='gray')
plt.colorbar()
plt.title("Raw Chest X-RAY Image")

print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height, one single color channel.")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")

sns.displot(generated_image.ravel(),
            label=f"Pixel mean {np.mean(generated_image):.4f} and & Standard Deviation {np.std(generated_image):.4f}", kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel("# Pixels in Image")

#建立CNN model
#Class weights
weight_for_0=num_pneumonia/(num_normal+num_pneumonia)
weight_for_1=num_normal/(num_normal+num_pneumonia)

class_weight={0:weight_for_0, 1:weight_for_1}

print(f'weight for class 0 :{weight_for_0: .2f}')
print(f'weight for class 1 :{weight_for_1: .2f}')

#使用relu
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(180, 180, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(180, 180, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

r=model.fit(train,
           epochs=8,
           validation_data=validation,
           class_weight=class_weight,
           steps_per_epoch=5216//64,
           validation_steps=4)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='Val_loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(2,2,2)
plt.plot(r.history['accuracy'], label='Accuracy')
plt.plot(r.history['val_accuracy'], label='Val_accuracy')
plt.legend()
plt.title('Accuracy Evolution')


#evaluation (評估)
evaluation = model.evaluate(test)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

evaluation = model.evaluate(train)
print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report

pred=model.predict(test)

print(confusion_matrix(test.classes, pred>0.5))
pd.DataFrame(classification_report(test.classes, pred>0.5, 
                                   output_dict=True))


print(confusion_matrix(test.classes, pred>0.7))
pd.DataFrame(classification_report(test.classes, pred>0.7,
                                   output_dict=True))

# 使用 VGG16 模板

#from keras.models import Sequential
#from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16

vgg16_base_model=VGG16(input_shape=(180,180,3),
                       include_top=False,
                       weights='imagenet')

vgg16_base_model.summary()

vgg16_model= tf.keras.Sequential([
    vgg16_base_model,
    GlobalAveragePooling2D(),
    Dense(512,activation='relu'),
    BatchNormalization(),
    Dropout(0.6),
    Dense(128,activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64,activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1,activation='sigmoid')
])

opt=tf.keras.optimizers.Adam(learning_rate=0.001)
METRICS=['accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
        ]
vgg16_model.compile(optimizer=opt,loss='binary_crossentropy',
                    metrics=METRICS)

r=vgg16_model.fit(train,
                 epochs=10,
                 validation_data=validation,
                 class_weight=class_weight,
                 steps_per_epoch=100,
                 validation_steps=25)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(2, 2, 2)
plt.plot(r.history['accuracy'], label='Accuracy')
plt.plot(r.history['val_accuracy'], label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

#評估

evaluation =vgg16_model.evaluate(test)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

evaluation = vgg16_model.evaluate(train)
print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")

