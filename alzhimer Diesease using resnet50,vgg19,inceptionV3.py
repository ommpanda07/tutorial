#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# System libraries
from pathlib import Path
import os.path
from distutils.dir_util import copy_tree, remove_tree
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
import time

from PIL import Image
from random import randint

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
import itertools



from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow 

from tensorflow.keras.applications.resnet50 import ResNet50 # ResNet50


import scipy
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


# Set the seed value for experiment reproduci.bility.
seed = 1842
tensorflow.random.set_seed(seed)
np.random.seed(seed)
# Turn off warnings for cleaner looking notebook
warnings.simplefilter('ignore')


# In[2]:


#AUGMENTED DATA
aug_data = "C:/Users/ommpa/Downloads/dataseet/AugmentedAlzheimerDataset"
org_data = "C:/Users/ommpa/Downloads/dataseet/OriginalDataset"


# In[3]:


image_dir = Path(aug_data)

# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) 
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)


# In[4]:


def get_dataset(augmented = 0):
    if(augmented == 0):
        image_dir = Path(org_data)
    else:
        image_dir = Path(aug_data)
        
    # Get filepaths and labels
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) 
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)
    return image_df
        


# In[5]:


train_df = get_dataset(1)
test_df = get_dataset(0)


# In[6]:


pd.set_option('display.max_colwidth', None)
train_df.head()


# In[7]:


test_df.tail()


# # Random 25 pictures for Augmented Samples and Original Samples

# In[8]:


# Display 25 picture of the augmented dataset with their labels
random_index = np.random.randint(0, len(train_df), 25)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(12, 12),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(cv2.cvtColor(plt.imread(train_df.Filepath[random_index[i]]), cv2.COLOR_BGR2RGB))
    ax.set_title(train_df.Label[random_index[i]])
plt.tight_layout()
plt.savefig("Augmented.pdf")

plt.show()


# In[9]:


# Display 25 picture of the original dataset with their labels
random_index = np.random.randint(0, len(test_df), 25)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(12, 12),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(cv2.cvtColor(plt.imread(test_df.Filepath[random_index[i]]), cv2.COLOR_BGR2RGB))
    ax.set_title(test_df.Label[random_index[i]])
plt.tight_layout()
plt.savefig("Original.pdf")
plt.show()


# In[10]:


print("Train data set label distribution:\n",train_df.Label.value_counts())

print("\nTest data set label distribution:\n", test_df.Label.value_counts())


# In[11]:


base_dir = "/C:/Users/ommpa/Downloads/dataseet/"
root_dir = "./"
train_dir = base_dir + "AugmentedAlzheimerDataset/"
test_dir = base_dir + "OriginalDataset/"


# In[12]:


CLASSES = [ 'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented']

IMG_SIZE = 176
IMAGE_SIZE = [176, 176]
DIM = (IMG_SIZE, IMG_SIZE)


# In[13]:


import os

train_dir = r"C:\Users\ommpa\Downloads\dataseet\AugmentedAlzheimerDataset"  # Use a raw string

# Verify if the directory exists
if not os.path.exists(train_dir):
    print(f"Error: The directory '{train_dir}' does not exist.")
else:
    print("Directory found! Proceeding with data loading.")
    


# In[14]:


test_dir = "C:/Users/ommpa/Downloads/dataseet/OriginalDataset"

import os

if not os.path.exists(test_dir):
    print(f"Error: The directory '{test_dir}' does not exist.")
else:
    print("Directory found! Proceeding with data loading.")


# In[15]:


datagen = IDG(rescale = 1./255, validation_split=0.1)

train_gen = datagen.flow_from_directory(directory=train_dir,
                                             target_size=DIM,
                                             batch_size=400,
                                             class_mode='categorical',
                                             subset='training',
                                             shuffle=True)

validation_gen = datagen.flow_from_directory(directory=train_dir,
                                             target_size=DIM,
                                             batch_size=400,
                                             class_mode='categorical',
                                             subset='validation',
                                             shuffle=True)

test_gen = datagen.flow_from_directory(directory=test_dir,
                                             target_size=DIM,
                                             batch_size=6400,
                                             class_mode='categorical')


# In[16]:


test_gen_plot = datagen.flow_from_directory(directory=test_dir,
                                             target_size=DIM,
                                             batch_size=128,
                                             class_mode='categorical')


# In[17]:


def prepare_for_test(model, test_gen):
    data, y_true = test_gen.next()
    y_pred_ = model.predict(data, batch_size = 64)
    y_pred = []
    for i in range(y_pred_.shape[0]):
        y_pred.append(np.argmax(y_pred_[i]))
        
    y_true = np.argmax(y_true, axis=1)
    
    return y_true, y_pred


# ----

# ResNet50

# In[18]:


rn = ResNet50(input_shape=(176,176,3), weights='imagenet', include_top=False)
for layer in rn.layers:
    layer.trainable = False
x = Flatten()(rn.output)

prediction = Dense(4, activation='softmax')(x)

model = Model(inputs=rn.input, outputs=prediction)

model.compile(optimizer='adam',
loss=tensorflow.losses.CategoricalCrossentropy(),
metrics=[keras.metrics.AUC(name='auc'),'acc'])
callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=8,
                                            restore_best_weights=True)

tic = time.perf_counter()
history = model.fit(train_gen,
                    steps_per_epoch=len(train_gen),
                    validation_data=validation_gen,
                    validation_steps=len(validation_gen),
                    epochs=70, callbacks=callback)
# time
toc = time.perf_counter()
print("Total Time:{}".format(round((toc-tic)/60,2)))


# In[19]:


# Evaluate the model
results = model.evaluate(train_gen)

# Ensure you unpack the correct number of results based on your model
if len(results) == 2:
    train_loss, train_accuracy = results
    print(f'\nTrain loss: {train_loss:.2f}')
    print(f'Train Accuracy: {train_accuracy*100:.2f} %')
else:
    # If there are additional metrics, handle them
    print(f'\nTrain results: {results}')


# In[20]:


# Evaluate the model
results = model.evaluate(validation_gen)

# Ensure you unpack the correct number of results based on your model
if len(results) == 2:
    train_loss, train_accuracy = results
    print(f'\nValidation loss: {train_loss:.2f}')
    print(f'Validation Accuracy: {train_accuracy*100:.2f} %')
else:
    # If there are additional metrics, handle them
    print(f'\nValidation results: {results}')


# In[21]:


# Evaluate the model
results = model.evaluate(test_gen_plot)

# Ensure you unpack the correct number of results based on your model
if len(results) == 2:
    train_loss, train_accuracy = results
    print(f'\nTest loss: {train_loss:.2f}')
    print(f'Test Accuracy: {train_accuracy*100:.2f} %')
else:
    # If there are additional metrics, handle them
    print(f'\nTest results: {results}')


# In[22]:


def plot_training_metrics(train_hist, model, test_gen_plot, y_actual, y_pred, classes, model_name):
    
    # Evaluate the results:
    test_metrics = model.evaluate(test_gen_plot, verbose = False)
    AUC       = test_metrics[1]*100
    Acc       = test_metrics[2]*100 
    results_title =(f"\n Model AUC {AUC:.2f}%, Accuracy {Acc:.2f}% on Test Data\n")
    print(results_title.format(AUC, Acc))


# In[24]:


def plot_training_metrics(train_hist, model, test_gen_plot, y_actual, y_pred, classes, model_name):
    
    # Evaluate the results:
    test_metrics = model.evaluate(test_gen_plot, verbose = False)
    AUC       = test_metrics[1]*100
    Acc       = test_metrics[2]*100 
    results_title =(f"\n Model AUC {AUC:.2f}%, Accuracy {Acc:.2f}% on Test Data\n")
    print(results_title.format(AUC, Acc))

    
    # print classification report
    print(classification_report(y_actual, y_pred, target_names=classes))

    # extract data from training history for plotting
    history_dict    = train_hist.history
    loss_values     = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    auc_values      = history_dict['auc']
    val_auc_values  = history_dict['val_auc']
    epochs          = range(1, len(history_dict['auc']) + 1)

    # get the min loss and max accuracy for plotting
    max_auc = np.max(val_auc_values)
    min_loss = np.min(val_loss_values)
    
    # create plots
    plt.subplots(figsize=(12,4))
    
    # plot loss by epochs
    plt.subplot(1,3,1)
    plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss_values, 'cornflowerblue', label = 'Validation loss')
    plt.title('Validation Loss by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.axhline(y=min_loss,color='darkslategray', linestyle='--')
    plt.legend()

    # plot accuracy by epochs
    plt.subplot(1,3,2)
    plt.plot(epochs, auc_values, 'bo',label = 'Training AUC')
    plt.plot(epochs, val_auc_values, 'cornflowerblue', label = 'Validation AUC')
    plt.plot(epochs,[AUC/100]*len(epochs),'darkmagenta',linestyle = '--',label='Test AUC')
    plt.title('Validation AUC by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.axhline(y=max_auc,color='darkslategray', linestyle='--')
    plt.legend()

    
     # calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # create confusion matrix plot
    plt.subplot(1,3,3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.BuPu)
    plt.title(f"Confusion Matrix \nAUC: {AUC:.2f}%")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # loop through matrix, plot each 
    threshold = cm.max() / 2.
    for r, c in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(c, r, format(cm[r, c], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[r, c] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{model_name}.pdf")

    plt.show()


# In[26]:


import numpy as np

def prepare_for_test(model, test_gen):
    data, y_true = next(test_gen)  # Fetch the next batch using Python's next()
    y_pred_ = model.predict(data, batch_size=64)
    y_pred = [np.argmax(pred) for pred in y_pred_]  # Convert predictions to class indices
    return y_true, y_pred

# Example usage (assuming test_gen and model are defined):
# y_true, y_pred = prepare_for_test(model, test_gen)
# plot_training_metrics(history, model, test_gen_plot, y_true, y_pred, ['mild', 'moderate', 'normal', 'very-mild'], model_name="resnet50")


# In[27]:


y_true, y_pred = prepare_for_test(model, test_gen)
plot_training_metrics(history, model, test_gen_plot, y_true, y_pred, ['mild','moderate','normal','very-mild'], model_name = "resnet50")


# In[28]:


import numpy as np
from sklearn.metrics import classification_report

def prepare_for_test(model, test_gen):
    data, y_true = next(test_gen)  # Fetch the next batch
    y_pred_ = model.predict(data, batch_size=64)  # Raw predictions (probabilities)
    
    # Convert y_true from one-hot (multilabel-indicator) to class indices (multiclass)
    y_true_classes = np.argmax(y_true, axis=1)  # Assuming y_true is one-hot encoded
    # Convert y_pred from probabilities to class indices
    y_pred_classes = np.argmax(y_pred_, axis=1)
    
    return y_true_classes, y_pred_classes

# Example usage with your plot_training_metrics function
# Assuming test_gen is defined and model is trained
y_true, y_pred = prepare_for_test(model, test_gen)
print(classification_report(y_true, y_pred, target_names=['mild', 'moderate', 'normal', 'very-mild']))

# Call your plotting function
# plot_training_metrics(history, model, test_gen_plot, y_true, y_pred, ['mild', 'moderate', 'normal', 'very-mild'], model_name="resnet50")


# In[30]:


# SAVE MODEL
model_dir = "./alzheimer_resnet50_model.h5"  # Specify .h5 extension in the filepath
model.save(model_dir)  # Remove save_format argument
np.save('my_resnet50_history.npy', history.history)

# Optional: Load history later if needed
# history = np.load('my_resnet50_history.npy', allow_pickle=True).item()


# # vgg19

# In[32]:


from tensorflow.keras.applications import VGG19  # Import VGG19 from Keras applications
from tensorflow.keras.layers import Flatten

# Define the VGG19 model
vgg = VGG19(input_shape=(176, 176, 3), weights='imagenet', include_top=False)

# Freeze the layers of VGG19
for layer in vgg.layers:
    layer.trainable = False

# Add a Flatten layer to the output
x = Flatten()(vgg.output)


# In[33]:


vgg = VGG19(input_shape=(176,176,3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)

prediction = Dense(4, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
model.summary()


# In[34]:


model.compile(optimizer='adam',
loss=tensorflow.losses.CategoricalCrossentropy(),
metrics=[keras.metrics.AUC(name='auc'),'acc'])
callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=8,
                                            restore_best_weights=True)

tic = time.perf_counter()
history = model.fit(train_gen,
                    steps_per_epoch=len(train_gen),
                    validation_data=validation_gen,
                    validation_steps=len(validation_gen),
                    epochs=50, callbacks=callback)
# time
toc = time.perf_counter()
print("Total Time:{}".format(round((toc-tic)/60,2)))


# In[36]:


y_true, y_pred = prepare_for_test(model, test_gen)
plot_training_metrics(history, model, test_gen_plot, y_true, y_pred, ['mild','moderate','normal','very-mild'], model_name = "vgg19")


# In[37]:


import numpy as np
from sklearn.metrics import classification_report

def prepare_for_test(model, test_gen):
    data, y_true = next(test_gen)  # Fetch the next batch
    y_pred_ = model.predict(data, batch_size=64)  # Raw predictions (probabilities)
    
    # Convert y_true from one-hot (multilabel-indicator) to class indices (multiclass)
    y_true_classes = np.argmax(y_true, axis=1)  # Assuming y_true is one-hot encoded
    # Convert y_pred from probabilities to class indices
    y_pred_classes = np.argmax(y_pred_, axis=1)
    
    return y_true_classes, y_pred_classes

# Example usage with your plot_training_metrics function
# Assuming test_gen is defined and model is trained
y_true, y_pred = prepare_for_test(model, test_gen)
print(classification_report(y_true, y_pred, target_names=['mild', 'moderate', 'normal', 'very-mild']))

# Call your plotting function
# plot_training_metrics(history, model, test_gen_plot, y_true, y_pred, ['mild', 'moderate', 'normal', 'very-mild'], model_name="resnet50")


# In[39]:


# SAVE MODEL
model_dir = "./alzheimer_vgg19_model.h5"  # or use .keras extension
model.save(model_dir)                     # remove save_format argument
np.save('my_vgg19_history.npy', history.history)


# # inceptionV3

# In[41]:


from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Flatten

model = InceptionV3(input_shape=(176,176,3), weights='imagenet', include_top=False)

for layer in model.layers:
    layer.trainable = False

x = Flatten()(model.output)


# In[42]:


model = InceptionV3(input_shape=(176,176,3), weights='imagenet', include_top=False)
for layer in model.layers:
    layer.trainable = False
x = Flatten()(model.output)

prediction = Dense(4, activation='softmax')(x)

model = Model(inputs=model.input, outputs=prediction)

model.compile(optimizer='adam',
loss=tensorflow.losses.CategoricalCrossentropy(),
metrics=[keras.metrics.AUC(name='auc'),'acc'])
callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=8,
                                            restore_best_weights=True)

tic = time.perf_counter()
history = model.fit(train_gen,
                    steps_per_epoch=len(train_gen),
                    validation_data=validation_gen,
                    validation_steps=len(validation_gen),
                    epochs=50, callbacks=callback)
# time
toc = time.perf_counter()
print("Total Time:{}".format(round((toc-tic)/60,2)))


# In[43]:


y_true, y_pred = prepare_for_test(model, test_gen)
plot_training_metrics(history, model, test_gen_plot, y_true, y_pred, ['mild','moderate','normal','very-mild'], model_name = "inceptionv3")


# In[44]:


import numpy as np
from sklearn.metrics import classification_report

def prepare_for_test(model, test_gen):
    data, y_true = next(test_gen)  # Fetch the next batch
    y_pred_ = model.predict(data, batch_size=64)  # Raw predictions (probabilities)
    
    # Convert y_true from one-hot (multilabel-indicator) to class indices (multiclass)
    y_true_classes = np.argmax(y_true, axis=1)  # Assuming y_true is one-hot encoded
    # Convert y_pred from probabilities to class indices
    y_pred_classes = np.argmax(y_pred_, axis=1)
    
    return y_true_classes, y_pred_classes

# Example usage with your plot_training_metrics function
# Assuming test_gen is defined and model is trained
y_true, y_pred = prepare_for_test(model, test_gen)
print(classification_report(y_true, y_pred, target_names=['mild', 'moderate', 'normal', 'very-mild']))

# Call your plotting function
# plot_training_metrics(history, model, test_gen_plot, y_true, y_pred, ['mild', 'moderate', 'normal', 'very-mild'], model_name="resnet50")


# In[45]:


# SAVE MODEL
model_dir = "./alzheimer_inceptionv3_model.h5"  # or use .keras extension
model.save(model_dir)                     # remove save_format argument
np.save('my_inceptionv3_history.npy', history.history)


# In[ ]:




