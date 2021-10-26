# Import Librarires
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, InceptionV3, NASNetMobile, NASNetLarge, DenseNet121, ResNet50, Xception, InceptionResNetV2
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,  EarlyStopping
from tensorflow.keras.models import Model

import cv2
from skimage import exposure

# Predefined models

models_ = dict(
                VGG16 = dict(
                    input_shape = (224,224,3),
                    module_name = "vgg16",
                    last_conv_layer = "block5_conv3",
                ),
                VGG19 = dict(
                    input_shape = (224,224,3),
                    module_name = "vgg19",
                    last_conv_layer = "block5_conv4",
                ),
                # this is used for ChexNet
                DenseNet121=dict(
                    input_shape=(224, 224, 3),
                    module_name="densenet",
                    last_conv_layer="conv5_block16_concat",
                ),
                ResNet50=dict(
                    input_shape=(224, 224, 3),
                    module_name="resnet50",
                    last_conv_layer="conv5_block3_out",
                ),
                InceptionV3=dict(
                    input_shape=(299, 299, 3),
                    module_name="inception_v3",
                    last_conv_layer="mixed10",
                ),
                InceptionResNetV2=dict(
                    input_shape=(299, 299, 3),
                    module_name="inception_resnet_v2",
                    last_conv_layer="conv_7b_bn",
                ),
                NASNetMobile=dict(
                    input_shape=(224, 224, 3),
                    module_name="nasnet",
                    last_conv_layer="normal_concat_12",
                ),
                NASNetLarge=dict(
                    input_shape=(331, 331, 3),
                    module_name="nasnet",
                    last_conv_layer="normal_concat_18",
                ),
                Xception=dict(
                    input_shape=(299, 299, 3),
                    module_name="xception",
                    last_conv_layer="block14_sepconv2_act",
                ),
            
            )


###################  Configuration   ##########################

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
    
isKaggleData = True  # purpose of kaggle running

classification_type = "binary"   # multi or binary
classifier = "ann"  # ann or svm

 # training images directory

#  Feature extract for ML classifiers (SVM)
train_num = 20  # that means below generator yields number of train images = train_num * batch_size
val_num = 6
show_cv_split_values = True  # true or false
feature_number = 128


use_fine_tuning = True   # if transfer model's weights are trainable
use_chex_weights = True  # use chexnet weights


###########################################################################

if isKaggleData:
    df_image = pd.read_csv('../input/siim-covid19-detection/train_image_level.csv')
    df_study = pd.read_csv('../input/siim-covid19-detection/train_study_level.csv')
    df_study['id'] = df_study['id'].str.replace('_study',"")
    df_study.rename({'id': 'StudyInstanceUID'},axis=1, inplace=True)
    df_train = df_image.merge(df_study, on='StudyInstanceUID')
    df_train.loc[df_train['Negative for Pneumonia']==1, 'study_label'] = 'negative'
    df_train.loc[df_train['Typical Appearance']==1, 'study_label'] = 'typical'
    df_train.loc[df_train['Indeterminate Appearance']==1, 'study_label'] = 'indeterminate'
    df_train.loc[df_train['Atypical Appearance']==1, 'study_label'] = 'atypical'
    df_train.drop(['Negative for Pneumonia','Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance'], axis=1, inplace=True)
    df_train['id'] = df_train['id'].str.replace('_image', '.jpg')
    df_train['image_label'] = df_train['label'].str.split().apply(lambda x : x[0])
    df_size = pd.read_csv('../input/covid-jpg-512/size.csv')
    data = df_train.merge(df_size, on='id')
    data = data.drop(["boxes","label","StudyInstanceUID","dim0","dim1","split"], axis = 1)
    img_dir = "../input/covid-jpg-512/train"
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

# Use these functions if image preprocessing required before training
# =============================================================================

def img_rescale(img):
    img_rescaled = exposure.rescale_intensity(img)
    return img_rescaled

def img_equalize(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq

def img_adapt_eq(img):
    img_adap = exposure.equalize_adapthist(img/255, clip_limit=0.03, kernel_size=24)
    return img_adap
# =============================================================================
 

image_generator_train = ImageDataGenerator(
            validation_split=0.2,
            #rotation_range=20,
            horizontal_flip = True,
            zoom_range = 0.1,
            #shear_range = 0.1,
            brightness_range = [0.8, 1.1],
            fill_mode='nearest',
            preprocessing_function=img_adapt_eq
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

def get_last_conv_layer(model_name):
    layer = base_model.get_layer(models_[model_name]["last_conv_layer"])
    return layer

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
        
    def generate_images_for_SVM(train_num, val_num):

        x_list = []
        y_list = []
        for i in range(train_num):
            x, y = next(train_generator)
            x_list.append(x)
            y_list.append(y)
            
        args = (x_list[i] for i in range(train_num))
        x_tr = np.vstack((args))
        args = (y_list[i] for i in range(train_num))
        y_tr = np.vstack(args)
        y_tr = y_tr.ravel()
            
        x_list = []
        y_list = []
        for i in range(val_num):
            x, y = next(valid_generator)
            x_list.append(x)
            y_list.append(y)
                
        args = (x_list[i] for i in range(val_num))
        x_val = np.vstack((args))
        args = (y_list[i] for i in range(val_num))
        y_val = np.vstack(args)
        y_val = y_val.ravel()
            
        return x_tr, x_val, y_tr, y_val
        
    #  Feature extract for ML classifiers

    x_tr, x_val, y_tr, y_val = generate_images_for_SVM(train_num, val_num)
    
    #  Feature extract for ML classifiers
    
    def extract_features_from_images(x_tr, x_val, y_tr, y_val):
        
        x_train = model.predict(x_tr)
        x_test = model.predict(x_val)
        y_train = y_tr
        y_test = y_val
        
        return x_train, x_test, y_train, y_test
    
    x_train, x_test, y_train, y_test = extract_features_from_images(x_tr, x_val, y_tr, y_val)
    

    print("Extracted data shapes from transfer network")
    print("x_train shape: ",x_train.shape)
    print("x_test shape: ",x_test.shape)
    print("y_train shape: ",y_train.shape)
    print("y_test shape: ",y_test.shape)
    
    # SVM tuning
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    
    
    svc_param_grid = {"kernel" : ["rbf", "poly", "linear"],
                      "gamma": [0.001, 0.01, 0.1, 1],
                      "C": [1,10,50,100,200,300]}
    
    clf = GridSearchCV(SVC(random_state = 42), param_grid = svc_param_grid, 
                       cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", 
                       n_jobs = -1,verbose = 1)
    clf.fit(x_train, y_train)
    print("Best score: ", clf.best_score_)
    print("Best estimator: ", clf.best_estimator_)
    
    
    # Prediction and Confusion Matrix result
    svc = clf.best_estimator_
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    train_accuracy = svc.score(x_train, y_train)
    test_accuracy = svc.score(x_test, y_test) 
    print("SVM Train accuracy",train_accuracy)
    print("SVM Test Accuracy",test_accuracy)
    
    plot_cv_scores(max_rank = 10)
    
    if show_cv_split_values:
        cross_val_score_plot(number_of_top = 7)
    
    get_confusion_matrix()
    
def plot_cv_scores(max_rank = 10):
    cv_results = pd.DataFrame(clf.cv_results_)
    cv_results = cv_results[
        ['params', 'rank_test_score', 'mean_test_score', 'std_test_score']
    ]
    cv_results = cv_results[cv_results['rank_test_score'] < max_rank]
    
    cv_results = (
        cv_results
        .set_index(cv_results["params"].apply(
            lambda x: "_".join(str(val) for val in x.values()))
        )
        .rename_axis('kernel')
    )
    
    plt.figure(figsize = (10,8))
    sns.lineplot(data=cv_results['mean_test_score'])
    plt.xticks(rotation = 90)
    plt.xlabel("Parameters")
    plt.ylabel("Mean test score")
    plt.show()
    
def get_confusion_matrix():
    
    if classifier == "ann":
        actual =  valid_generator.labels
        preds = np.argmax(model.predict(valid_generator), axis=1)
        cfmx = confusion_matrix(actual, preds)
        acc = accuracy_score(actual, preds)
        
        print ('Test Accuracy:', acc )
        heatmap(cfmx, annot=True, cmap='plasma',
            xticklabels=list(valid_generator.class_indices.keys()),
                fmt='.0f', 
                yticklabels=list(valid_generator.class_indices.keys())
                )
        plt.show()
    else:
        actual =  y_test
        preds = svc.predict(x_test)
        cfmx = confusion_matrix(actual, preds)
        acc = accuracy_score(actual, preds)
        
        print ('Test Accuracy:', acc )
        heatmap(cfmx, annot=True, cmap='plasma',
            xticklabels=list(np.unique(y_test)),
                fmt='.0f', 
                yticklabels=list(np.unique(y_test))
                )
        
        
def get_confusion_matrix():
    
    if classifier == "ann":
        actual =  valid_generator.labels
        preds = np.argmax(model.predict(valid_generator), axis=1)
        cfmx = confusion_matrix(actual, preds)
        acc = accuracy_score(actual, preds)
        
        print ('Test Accuracy:', acc )
        heatmap(cfmx, annot=True, cmap='plasma',
            xticklabels=list(valid_generator.class_indices.keys()),
                fmt='.0f', 
                yticklabels=list(valid_generator.class_indices.keys())
                )
        plt.show()
    else:
        actual =  y_test
        preds = svc.predict(x_test)
        cfmx = confusion_matrix(actual, preds)
        acc = accuracy_score(actual, preds)
        
        print ('Test Accuracy:', acc )
        heatmap(cfmx, annot=True, cmap='plasma',
            xticklabels=list(np.unique(y_test)),
                fmt='.0f', 
                yticklabels=list(np.unique(y_test))
                )
        
def plot_tl_metrics():
    
    hist = pd.DataFrame(history.history)
    hist.index += 1
        
    fig, (ax1, ax2) = plt.subplots(figsize=(12,12),nrows=2, ncols=1)
    hist['loss'].plot(ax=ax1,c='k',label='training loss')
    hist['val_loss'].plot(ax=ax1,c='r',linestyle='--', label='validation loss')
    ax1.legend()
    
    hist['accuracy'].plot(ax=ax2,c='k',label='training accuracy')
    hist['val_accuracy'].plot(ax=ax2,c='r',linestyle='--',label='validation accuracy')
    ax2.legend()
    plt.suptitle(f"{model_name} Loss and Accuracy Plots")
    plt.show()
    
def cross_val_score_plot(number_of_top = 7):
    results_df = pd.DataFrame(clf.cv_results_)
    results_df = results_df.sort_values(by=['rank_test_score'])
    results_df = (
        results_df
        .set_index(results_df["params"].apply(
            lambda x: "_".join(str(val) for val in x.values()))
        )
        .rename_axis('kernel')
    )
    
    # create df of model scores ordered by performance
    model_scores = results_df.filter(regex=r'split\d*_test_score')
    model_scores = model_scores.transpose().iloc[:30,:number_of_top]

    # plot 30 examples of dependency between cv fold and AUC scores
    fig, ax = plt.subplots(figsize = (8,12))
    sns.lineplot(
        data=model_scores,
        dashes=False, palette='Set1', marker='o', alpha=.5, ax=ax
    )
    ax.set_xlabel("CV test fold", size=12, labelpad=10)
    ax.set_ylabel("Model AUC", size=12)
    ax.tick_params(bottom=True, labelbottom=False)
    plt.show()
