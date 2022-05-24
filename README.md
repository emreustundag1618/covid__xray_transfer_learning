# Covid-19 Classification from Chest X-Ray Images with Deep Transfer Learning Methods

## Emre Üstündağ, M.Sc Thesis

### Graduate School of Natural and Applied Sciences - Department of Statistics

### **Summary**

This application has been prepared as a master's thesis for the classification of Covid-19 disease from chest X-ray images. Convolutional neural networks and support vector machines are used for classification tasks.

### **Classes with radiological findings:**

*Negative:* Pneumonia negative, no obvious opacity, healthy appearance

*Typical appearance:* Multifocal, bilateral, peripheral opacities, opacities with rounded morphology, lower lung-dominant distribution

*Atypical appearance:* Absence of typical findings and unilateral opacities, central or upper lung predominant distribution

*Indeterminate appearance:* Findings that doesn't belong to typical or atypical appearances. For example, pneumothorax or pleural effusion, pulmonary edema, lobar consolidation, solitary lung nodule or mass, diffuse small nodules, cavity

### **Classification types:**

*Negative - Positive* with all appearence types (binary)

*Negative – Typical* (binary)

*Negative – Atypical* (binary)

*Negative – Indeterminate* (binary)

*Negative - Typical - Atypical – Indeterminate* (multi)

#### Dataset used in the study:

[https://www.kaggle.com/c/siim-covid19-detection](https://www.kaggle.com/c/siim-covid19-detection)

For more info about the dataset: [https://osf.io/532ek/](https://osf.io/532ek/)

### Configuration:

model\_name: CNN model name. &quot;DenseNet121&quot;, &quot;ResNet50&quot;, &quot;InceptionV3&quot; or &quot;Xception&quot;.

learning\_rate: Learning rate for Adam optimizer. Default is 1e-8

min\_learning\_rate: Minimum learning rate when using ReduceLROnPlateau callback function

batch\_size: Size of batch size.

epochs: Number of epochs for CNN training

verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action

img\_process\_function: Image preprocessing function type from skimage library.

- For stretching or shrinking images&#39; intensity levels use &quot;rescale\_intensity&quot;
- For histogram equalization use &quot;equalize\_hist&quot;
- For Contrast Limited Adaptive Histogram Equalization (CLAHE) use &quot;equalize\_adapthist&quot;

isKaggleData: To train on Kaggle make this True and add following datasets to your Kaggle notebook:

- [https://www.kaggle.com/c/siim-covid19-detection](https://www.kaggle.com/c/siim-covid19-detection)
- [https://www.kaggle.com/datasets/sinamhd9/chexnet-weights](https://www.kaggle.com/datasets/sinamhd9/chexnet-weights)
- [https://www.kaggle.com/datasets/sinamhd9/covid-jpg-512](https://www.kaggle.com/datasets/sinamhd9/covid-jpg-512)
- [https://www.kaggle.com/datasets/emreustundag/dropped-siim](https://www.kaggle.com/datasets/emreustundag/dropped-siim)

You can also reach kaggle notebooks used in this study:

- CNN notebook: [https://www.kaggle.com/code/emreustundag/siim-covid-tl-ann-classifier](https://www.kaggle.com/code/emreustundag/siim-covid-tl-ann-classifier)
- CNN + SVM notebook: [https://www.kaggle.com/code/emreustundag/siim-covid-tl-ann-classifier](https://www.kaggle.com/code/emreustundag/siim-covid-tl-ann-classifier)

classification\_type: Classification type &quot;binary&quot; or &quot;multi&quot;.

use\_fine\_tuning: True if wanted to use fine tuning to update convolutional layers.

use\_chex\_weights: True if ChexNet weights are used instead of DenseNet121.

show\_versions: True if wanted to show libraries used with versions.

show\_model\_summary: True if wanted to display CNN model summary.

save\_weights: True if wanted to save trained model.

classes: Binary classification subtypes .

- For typical and negative, use: &quot;typical-none&quot;
- For atypical and negative, use: &quot;atypical-none&quot;
- For indeterminate and negative, use: &quot;indeterminate -none&quot;
- For all positive apperances and negative, use: &quot;all&quot;

#### **For SVM classifier**

train\_num: Number of image data for SVM. &quot;5500 / batch\_size&quot; means 5500 training images&#39; features will be extracted from CNN model.

val\_num = Number of image data for SVM. &quot;800 / batch\_size&quot; means 800 test images&#39; features will be extracted from CNN model.

show\_cv\_scores = True if wanted to plot cross-validation accuracy scores.

feature\_number: Number of features extracted from CNN model for SVM. 1000 has been used in the study.

svm\_hyp\_search: Hyperparameter optimization technique for SVM.

### Results:

| **Classification Type** | **CNN Model** | **CNN** | **CNN + SVM** |
| --- | --- | --- | --- |
| Binary: Covid-19 (+) or (-) | ResNet50V2 </br>InceptionV3</br>XCeption</br>CheXNet | 0.7850</br>0.7960</br>0.800</br> *0.807* | 0.7180</br>0.7990</br>0.7220</br>0.794 |
| Multi: Typical, Atypical, Indeterminate, Covid-19(-) | ResNet50V2</br>InceptionV3</br>XCeption</br>CheXNet | 0.7470</br>0.7500</br>0.744</br> *0.771* | 0.6750</br>0.7260</br>0.6990</br>0.716 |
| Binary: Typical appearence or Covid-19 (-) | ResNet50V2</br>InceptionV3</br>XCeption</br>CheXNet | 0.8080</br>0.8350</br>0.8330</br>0.846 | 0.779 </br>*0.856* </br>0.8400</br>0.832 |
| Binary: Atypical appearence or Covid-19 (-) | ResNet50V2</br>InceptionV3</br>XCeption</br>CheXNet | 0.5000</br>0.5800</br>0.5310</br>0.645 | 0.5000</br>0.6360</br>0.582</br> *0.683* |
| Binary: Indeterminate appearence or Covid-19 (-) | ResNet50V2</br>InceptionV3</br>XCeption</br>CheXNet | 0.6240</br>0.6860</br>0.5340</br>0.698 | 0.6200</br>0.6600</br>0.689 </br>*0.752* |

