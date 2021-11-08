# %% Import Librarires

# custom modules
from functions import *
from transfer_models import *

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,  EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configparser import ConfigParser
import importlib
from skimage import exposure

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

libraries = cp["DEFAULT"].get("libraries").split(",")
show_versions = cp["DEFAULT"].getboolean("show_versions")

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

# Defined image preprocessing functions

def preprocessing_function(img):
    
    global img_process_function
    func = img_process_function
    
    if func == "equalize_adapthist":
        img = exposure.equalize_adapthist(img/255, clip_limit=0.03, kernel_size=24)
    elif func == "equalize_hist":
        img = exposure.equalize_hist(img/255, clip_limit=0.03, kernel_size=24)
    elif func == "rescale_intensity":
        img = exposure.rescale_intensity(img/255, clip_limit=0.03, kernel_size=24)
        
    return img



image_generator_train = ImageDataGenerator(
            validation_split=0.2,
            #rotation_range=20,
            horizontal_flip = True,
            zoom_range = 0.1,
            #shear_range = 0.1,
            brightness_range = [0.8, 1.1],
            fill_mode='nearest',
            preprocessing_function=preprocessing_function)
    
image_generator_valid = ImageDataGenerator(validation_split=0.2,
                                           preprocessing_function=preprocessing_function)
    
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
    x = get_last_conv_layer(base_model, model_name).output
    output = GlobalAveragePooling2D()(x)
    
else:
    x = get_last_conv_layer(base_model, model_name).output
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
        
        
    # Keras callbacks
    rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, verbose = verbose, 
                                    min_delta = 1e-4, min_lr = min_learning_rate, mode = 'min')
    
    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min', 
                        restore_best_weights = True, verbose = verbose)
    
    ckp = ModelCheckpoint('model.h5',monitor = 'val_loss',
                          verbose = 0, save_best_only = True, mode = 'min')
    
    # Model fitting
    history = model.fit(
          train_generator,
          epochs= epochs,
          validation_data=valid_generator,
          callbacks=[es, rlr, ckp],
          verbose= verbose
          )
    
    if use_fine_tuning:
        model.save_weights(f"{model_name}-model.h5")
        
    plot_tl_metrics(history, model_name)
    
    plot_tl_confusion_matrix(model, valid_generator)
        

# %% X train and test extractions for ML classifiers, e.g SVM

if classifier == "svm":
    
    output = Dense(feature_number, activation = "relu", name = "features")(output)
    model = Model(base_model.input, output)
    
    #  Generate train and test images from generators
    x_tr, x_val, y_tr, y_val = generate_images_for_SVM(train_generator, valid_generator, train_num, val_num)
    # Extract feature vectors
    x_train, x_test, y_train, y_test = extract_features_from_images(model, x_tr, x_val, y_tr, y_val)
    # Print feature vectors' shapes
    print_feature_shapes(x_train, x_test, y_train, y_test)  
    # Fit SVM cv models
    clf, svc, y_pred = fit_cross_models(x_train, x_test, y_train, y_test)
    # Print best estimators and accuracy scores
    print_best_results(clf, svc, x_train, y_train, x_test, y_test)
    

    if show_cv_scores:
        plot_cv_splits(clf, number_of_top = 7)
        plot_cv_scores(clf, max_rank = 10)
    
    plot_svm_confusion_matrix(svc, x_test, y_test)