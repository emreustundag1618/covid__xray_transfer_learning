import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from skimage import exposure
from transfer_models import get_models


# Returns model's last convolution layer

def get_last_conv_layer(base_model, model_name):
    
    models_ = get_models()
    layer = base_model.get_layer(models_[model_name]["last_conv_layer"])
    
    return layer

# Use this for Covid TL models on Kaggle: https://www.kaggle.com/c/siim-covid19-detection

def prepare_data_for_kaggle():
    
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
    
    return data, img_dir


     
# If the classifier is ANN, this function plots transfer models' history

def plot_tl_metrics(history, model_name):
    
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



# Generates images for SVM classifier

def prepare_images_for_SVM(train_generator, valid_generator, train_num, val_num):
    
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


# Extracts features from generated images for SVM classifier 
    
def extract_features_from_images(model, x_tr, x_val, y_tr, y_val):
        
    x_train = model.predict(x_tr)
    x_test = model.predict(x_val)
    y_train = y_tr
    y_test = y_val
        
    return x_train, x_test, y_train, y_test


# Prints feature vectors' shapes
    
def print_feature_shapes(x_train, x_test, y_train, y_test):
    
    print("Extracted data shapes from transfer network")
    print("x_train shape: ",x_train.shape)
    print("x_test shape: ",x_test.shape)
    print("y_train shape: ",y_train.shape)
    print("y_test shape: ",y_test.shape)
    
    

# Fits SVM model with Grid Search or Bayes Search Cross Validation   

def fit_cross_models(x_train, x_test, y_train, y_test, svm_hyp_search):
    
    if svm_hyp_search == "grid":
        
        svc_param_grid = {"kernel" : ["rbf", "poly", "linear"],
                          "gamma": [0.001, 0.01, 0.1, 1],
                          "C": [1,10,50,100,200,300]}
        
        clf = GridSearchCV(SVC(random_state = 42), param_grid = svc_param_grid, 
                           cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", 
                           n_jobs = -1,verbose = 1)
        
        
        
    else:
        
        from skopt import BayesSearchCV
        
        # log-uniform: understand as search over p = exp(x) by varying x
        
        search_spaces = {
                        'C': (1e-4, 1e+4, 'log-uniform'),
                        'gamma': (1e-4, 1e+4, 'log-uniform'),
                        'degree': (1, 8),  # integer valued parameter
                        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
                        }
        
        clf = BayesSearchCV(
            SVC(random_state = 42),
            search_spaces=search_spaces,
            n_iter=32,
            cv=3)
        
        

    clf.fit(x_train, y_train)
        
    svc = clf.best_estimator_
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    
    return clf, svc, y_pred


# Plots each cross validation split's accuracy scores for number of top pairs of parameters

def plot_cv_splits(clf, number_of_top = 7):
    
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
    
    plt.subplots(figsize = (8,12))
    sns.lineplot(
        data=model_scores,
        dashes=False, palette='Set1', marker='o', alpha=.5
    )
    plt.xlabel("CV test fold")
    plt.ylabel("Model AUC")
    plt.xticks(rotation = 45)
    plt.show()
    

# TShows each combinations of parameter's accuracy scores on SVM, max_rank = 10 means "best ten combination"

def plot_cv_scores(clf, max_rank = 10):
    
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
    plt.xticks(rotation = 45)
    plt.xlabel("Parameters")
    plt.ylabel("Mean test score")
    plt.show()

    
# Prints best SVM scores and estimator
    
def print_best_results(clf, svc, x_train, y_train, x_test, y_test):
    
    train_accuracy = svc.score(x_train, y_train)
    test_accuracy = svc.score(x_test, y_test)
    
    print("Best SVM estimator (parameters): ", clf.best_estimator_)
    print("SVM Best Train accuracy",train_accuracy)
    print("SVM Best Test Accuracy",test_accuracy)
    
    
# Shows confusion matrix for ANN classifier
        
def plot_tl_confusion_matrix(model, valid_generator):
    
    actual =  valid_generator.labels
    preds = np.argmax(model.predict(valid_generator), axis=1)
    cfmx = confusion_matrix(actual, preds)
    acc = accuracy_score(actual, preds)
    
    sns.heatmap(cfmx, annot=True, cmap='plasma',
        xticklabels=list(valid_generator.class_indices.keys()),
            fmt='.0f', 
            yticklabels=list(valid_generator.class_indices.keys())
            )
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.show()


# Shows confusion matrix for SVM classifier
    
def plot_svm_confusion_matrix(svc, x_test, y_test):
    
    actual =  y_test
    preds = svc.predict(x_test)
    cfmx = confusion_matrix(actual, preds)
    acc = accuracy_score(actual, preds)    
    
    plt.figure()
    sns.heatmap(cfmx, annot=True, cmap='plasma',
        xticklabels=list(np.unique(y_test)),
            fmt='.0f', 
            yticklabels=list(np.unique(y_test))
            )
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.show()


# Displays imported libraries versions

def display_versions(libraries = None):
    
    from importlib import import_module
    
    for library in libraries:
        print(f"{library} version: {import_module(library).__version__}")


# Generates image data

def generate_images(classifier, classification_type, img_process_function, df_train, img_dir, img_size, batch_size):
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    from skimage import exposure
    
    # Defined image preprocessing functions

    def preprocess_function(img):
        
        if img_process_function == "equalize_adapthist":
            img = exposure.equalize_adapthist(img/255, clip_limit=0.03, kernel_size=24)
        elif img_process_function == "equalize_hist":
            img = exposure.equalize_hist(img/255, clip_limit=0.03, kernel_size=24)
        elif img_process_function == "rescale_intensity":
            img = exposure.rescale_intensity(img/255, clip_limit=0.03, kernel_size=24)
            
        return img
    
    
    
    # parameters for SVM classifier to be extracted from ImageDataGenerators
    
    featurewise_center=False
    samplewise_center=False
    featurewise_std_normalization=False
    samplewise_std_normalization=False
    zca_whitening=False
    rotation_range=0
    width_shift_range=0.0
    height_shift_range=0.0
    brightness_range=None
    shear_range=0.0
    zoom_range=0.0
    channel_shift_range=0.0
    cval=0.0
    horizontal_flip=False
    vertical_flip=False
    rescale=None
    preprocessing_function=preprocess_function
    validation_split=0.25

    
    if classification_type == "binary":
        y_col = "image_label"
    else:
        y_col = "study_label"

    if classifier == "ann":
        class_mode = "categorical"
        zca_epsilon=1e-06
        rotation_range=20
        width_shift_range=0.0
        height_shift_range=0.0
        brightness_range=[0.8, 1.1]
        shear_range=0.1
        zoom_range=0.0
        channel_shift_range=0.0
        fill_mode='nearest'
        cval=0.0
        horizontal_flip=False
        vertical_flip=False
        rescale=None
        preprocessing_function=preprocess_function
    else:
        class_mode = "raw"
        
    
    image_generator_train = ImageDataGenerator(
                featurewise_center=featurewise_center,
                samplewise_center=samplewise_center,
                featurewise_std_normalization=featurewise_std_normalization,
                samplewise_std_normalization=samplewise_std_normalization,
                zca_whitening=zca_whitening,
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                brightness_range=brightness_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                channel_shift_range=channel_shift_range,
                cval=cval,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                rescale=rescale,
                preprocessing_function=preprocessing_function,
                validation_split=validation_split)
        
    image_generator_valid = ImageDataGenerator(validation_split=validation_split,
                                               preprocessing_function=preprocessing_function)
        
    train_generator = image_generator_train.flow_from_dataframe(
                dataframe = df_train,
                directory=img_dir,
                x_col = 'id',
                y_col =  y_col,  
                target_size=(img_size, img_size),
                batch_size=batch_size,
                subset='training', 
                seed = 42, 
                class_mode = class_mode) 
        
    valid_generator = image_generator_valid.flow_from_dataframe(
            dataframe = df_train,
            directory=img_dir,
            x_col = 'id',
            y_col = y_col,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            subset='validation', 
            shuffle=False,  
            seed=42, 
            class_mode = class_mode)
    
    return train_generator, valid_generator
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
