# %% Import Librarires

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,  EarlyStopping
from tensorflow.keras.models import Model


from functions import *


# %% ###################  Configuration   ##########################

model_name = "DenseNet121"  # transfer model to use for data
learning_rate = 0.0001   # initialize learning rate
min_learning_rate = 1e-8   # learning rate doesnt decrease further
input_shape = models_[model_name]["input_shape"]  # input shape required for transfer models
img_size = input_shape[0]
batch_size = 32
epochs = 2
verbose = 1 # controls the amount of logging done during training and testing: 
    # 0 - none , 
    # 1 - reports metrics after each batch 
    # 2 - reports metrics after each epoch
    
img_process_function = "equalize_adapthist" 
    # defined functions: equalize_adapthist, equalize_hist, rescale_intensity
    
isKaggleData = True  # purpose of kaggle running

classification_type = "binary"   # multi or binary
classifier = "ann"  # ann or svm

 # training images directory

#  Feature extract for ML classifiers (SVM)
train_num = 20  # that means below generator yields number of train images = train_num * batch_size
val_num = 6
show_cv_split_values = True  # true or false
feature_number = 128   # length of feature vector


use_fine_tuning = True   # if transfer model's weights are trainable
use_chex_weights = True  # use chexnet weights



###########################################################################

if isKaggleData:
    data, img_dir = prepare_data_for_kaggle()
else:
    data = pd.read_csv("data.csv")
    img_dir = "images/train"
    
df_train = data.copy()

if classification_type == "binary":
    y_col = "image_label"
else:
    y_col = "study_label"

if classifier != "ann":
    class_mode = "raw"
else:
    class_mode = "categorical"

# %% Generate Images

image_generator_train = ImageDataGenerator(
            validation_split=0.2,
            #rotation_range=20,
            horizontal_flip = True,
            zoom_range = 0.1,
            #shear_range = 0.1,
            brightness_range = [0.8, 1.1],
            fill_mode='nearest',
            preprocessing_function=preprocessing_function
    )
    
image_generator_valid = ImageDataGenerator(validation_split=0.2,
                                           preprocessing_function=img_adapt_eq)
    
train_generator = image_generator_train.flow_from_dataframe(
            dataframe = df_train,
            directory=img_dir,
            x_col = 'id',
            y_col =  y_col,  
            target_size=(img_size, img_size),
            batch_size=batch_size,
            subset='training', seed = 23, class_mode = class_mode) 
    
valid_generator=image_generator_valid.flow_from_dataframe(
        dataframe = df_train,
        directory=img_dir,
        x_col = 'id',
        y_col = y_col,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        subset='validation', shuffle=False,  seed=23, class_mode = class_mode)

# %% Model call, training and evaluating

base_model_class = getattr(
    importlib.import_module(
        f"keras.applications.{models_[model_name]['module_name']}"
        ),
        model_name)
          
img_input = Input(shape = input_shape)
        
base_model = base_model_class(
            include_top = False,
            input_tensor = img_input,
            input_shape = input_shape,
            weights = "imagenet",
            pooling = "avg")
base_model.trainable = False

if use_fine_tuning:
    base_model.trainable = True


if (model_name == "DenseNet121") & use_chex_weights:
    chex_weights_path = '../input/chexnet-weights/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
    out = Dense(14, activation='sigmoid')(base_model.output)
    base_model = Model(inputs=base_model.input, outputs=out)
    base_model.load_weights(chex_weights_path)
    x = get_last_conv_layer(model_name).output
    output = GlobalAveragePooling2D()(x)
    
else:
    x = get_last_conv_layer(model_name).output
    output = GlobalAveragePooling2D()(x)
    

if classifier == "ann":
    if classification_type == "multi":
        predictions = Dense(len(df_train.study_label.unique()), activation = "softmax", name = "multi_predictions")(output)
        model = Model(base_model.input, predictions)
        model.compile(Adam(lr=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
    else:
        predictions = Dense(len(df_train.image_label.unique()), activation = "softmax", name = "binary_predictions")(output)
        model = Model(base_model.input, predictions)
        model.compile(Adam(lr=learning_rate),loss='binary_crossentropy',metrics=['accuracy'])
        

    rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, verbose = verbose, 
                                    min_delta = 1e-4, min_lr = min_learning_rate, mode = 'min')
    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min', 
                        restore_best_weights = True, verbose = verbose)
    ckp = ModelCheckpoint('model.h5',monitor = 'val_loss',
                          verbose = 0, save_best_only = True, mode = 'min')
    history = model.fit(
          train_generator,
          epochs= epochs,
          validation_data=valid_generator,
          callbacks=[es, rlr, ckp],
          verbose= verbose
          )
    
    if use_fine_tuning:
        model.save_weights(f"{model_name}-model.h5")
        
    plot_tl_metrics()
    get_confusion_matrix()
        

# X train and test extractions for ML classifiers, e.g SVM

if classifier == "svm":
    output = Dense(feature_number, activation = "relu", name = "features")(output)
    model = Model(base_model.input, output)
        
       
    #  Feature extract for ML classifiers

    x_tr, x_val, y_tr, y_val = generate_images_for_SVM(train_num, val_num)
    
    #  Feature extract for ML classifiers

    x_train, x_test, y_train, y_test = extract_features_from_images(x_tr, x_val, y_tr, y_val)
    
    print_feature_shapes(x_train, x_test, y_train, y_test)

    show_SVM_results(x_train, x_test, y_train, y_test)
    
    
    if show_cv_split_values:
        cross_val_score_plot(number_of_top = 7)
    
    get_confusion_matrix()
    

    
    
