# %% Import Librarires

# custom modules
from functions import *
from transfer_models import *

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,  EarlyStopping
from tensorflow.keras.models import Model

from configparser import ConfigParser
import importlib

import os

import warnings
warnings.filterwarnings("ignore")

# parser config
config_file = "config.ini"
cp = ConfigParser()
cp.read(config_file)

# default config
model_name = cp["DEFAULT"].get("model_name")
learning_rate = cp["DEFAULT"].getfloat("learning_rate")
min_learning_rate = cp["DEFAULT"].getfloat("min_learning_rate")
batch_size = cp["DEFAULT"].getint("batch_size")
epochs = cp["DEFAULT"].getint("epochs")
verbose = cp["DEFAULT"].getint("verbose")
img_process_function = cp["DEFAULT"].get("img_process_function")
isKaggleData = cp["DEFAULT"].getboolean("isKaggleData")
classification_type = cp["DEFAULT"].get("classification_type")
classifier = cp["DEFAULT"].get("classifier")
train_num = cp["DEFAULT"].getint("train_num")
val_num = cp["DEFAULT"].getint("val_num")
show_cv_scores = cp["DEFAULT"].getboolean("show_cv_scores")
feature_number = cp["DEFAULT"].getint("feature_number")
use_fine_tuning = cp["DEFAULT"].getboolean("use_fine_tuning")
use_chex_weights = cp["DEFAULT"].getboolean("use_chex_weights")
svm_hyp_search = cp["DEFAULT"].get("svm_hyp_search")

libraries = cp["DEFAULT"].get("libraries").split(",")
show_versions = cp["DEFAULT"].getboolean("show_versions")
save_weights = cp["DEFAULT"].getboolean("save_weights")

if show_versions:
    display_versions(libraries)

models_ = get_models()
input_shape = models_[model_name]["input_shape"]
img_size = input_shape[0]


if isKaggleData:
    data, img_dir = prepare_data_for_kaggle()
else:
    data = pd.read_csv("train_data.csv")
    img_dir = "images/train"
    
df_data = data.copy()

# drop images from dataframe not in images directory
files = os.listdir("images/train")

not_in_files_index = []

for file_id in df_data.id:
    if file_id in files:
        continue
    else:
        not_in_files_index.append(df_data[df_data["id"] == file_id].index[0])
        
df_data = df_data.drop(not_in_files_index, axis = 0)

# drop images that have unclear view
drop_df = pd.read_csv("dropped_image_IDs.csv") + ".jpg"

drop_index = []
for row in drop_df.values:
    drop_index.append(df_data[df_data["id"] == row[0]].index[0])
    
    
df_data = df_data.drop(drop_index, axis = 0)

# splitting images train and test
df_train = df_data.iloc[:5000]
df_test = df_data.iloc[5000:]


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



if (model_name == "DenseNet121") & use_chex_weights:
    
    chex_weights_path = 'brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
    out = Dense(14, activation='sigmoid')(base_model.output)
    base_model = Model(inputs=base_model.input, outputs=out)
    base_model.load_weights(chex_weights_path)
    x = get_last_conv_layer(base_model, model_name).output
    output = GlobalAveragePooling2D()(x)
    
else:
    x = get_last_conv_layer(base_model, model_name).output
    output = GlobalAveragePooling2D()(x)


base_model.trainable = False

if use_fine_tuning:   
    base_model.trainable = True
    
    
    print("Train and Validation Sets for Training Transfer Model:")
    
    train_generator, valid_generator = generate_images_for_model_training( classifier = classifier, 
                                                                           classification_type = classification_type, 
                                                                           img_process_function = img_process_function, 
                                                                           df_train = df_train, 
                                                                           df_test = df_test, 
                                                                           img_dir = img_dir, 
                                                                           img_size = img_size, 
                                                                           batch_size = batch_size, 
                                                                           validation_split = 0.15)
        
    if classification_type == "multi":
        predictions = Dense(len(df_train.study_label.unique()), activation = "softmax", name = "multi_predictions")(output)
        model = Model(base_model.input, predictions)
        model.compile(Adam(lr=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
        
    else:
        predictions = Dense(len(df_train.image_label.unique()), activation = "softmax", name = "binary_predictions")(output)
        model = Model(base_model.input, predictions)
        model.compile(Adam(lr=learning_rate),loss='binary_crossentropy',metrics=['accuracy'])
        
        
    # Keras callbacks
    rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, verbose = verbose, 
                                    min_delta = 1e-4, min_lr = min_learning_rate, mode = 'min')
    
    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min', 
                        restore_best_weights = True, verbose = verbose)
    
    ckp = ModelCheckpoint('model.h5',monitor = 'val_loss',
                          verbose = verbose, save_best_only = True, mode = 'min')
    
    # Model fitting
    history = model.fit(
          train_generator,
          epochs= epochs,
          validation_data=valid_generator,
          callbacks=[es, rlr, ckp],
          verbose= verbose
          )
    
    if save_weights:
        model.save_weights(f"{model_name}-model.h5")
        
    if classifier == "ann":
    
        plot_tl_metrics(history, model_name)
        
        plot_tl_confusion_matrix(model, valid_generator)
    

# %% X train and test extractions for ML classifiers, e.g SVM

if classifier == "svm":
    
    output = Dense(feature_number, activation=LeakyReLU(alpha=0.2), name = "features")(output)
    model = Model(base_model.input, output)
    
    print("Train and Test Sets for SVM Classifier:")
    # for svm classifier there are different generators
    train_generator, test_generator = generate_images_for_feature_extraction( classifier = classifier, 
                                                                               classification_type = classification_type, 
                                                                               img_process_function = img_process_function, 
                                                                               df_train = df_train, 
                                                                               df_test = df_test, 
                                                                               img_dir = img_dir, 
                                                                               img_size = img_size, 
                                                                               batch_size = batch_size 
                                                                               )
    
    

    #  Generate train and test images from generators
    x_tr, x_val, y_tr, y_val = prepare_images_for_SVM(train_generator, test_generator, train_num, val_num)
    # Extract feature vectors
    x_train, x_test, y_train, y_test = extract_features_from_images(model, x_tr, x_val, y_tr, y_val)
    # Print feature vectors' shapes
    print_feature_shapes(x_train, x_test, y_train, y_test)  
    # Fit SVM cv models
    clf, svc, y_pred = fit_cross_models(x_train, x_test, y_train, y_test, svm_hyp_search)
    # Print best estimators and accuracy scores
    print_best_results(clf, svc, x_train, y_train, x_test, y_test)
    
    if svm_hyp_search == "grid":
        if show_cv_scores:
            plot_cv_splits(clf, number_of_top = 7)
            plot_cv_scores(clf, max_rank = 10)
    
    plot_svm_confusion_matrix(svc, x_test, y_test)