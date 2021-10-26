import pandas as pd


image_path = 'train_image_level.csv'
study_path = 'train_study_level.csv'

def load_data(image_path, study_path):
      
    df_image = pd.read_csv(image_path)   
    df_study = pd.read_csv(study_path)
    
    return df_image, df_study
    

def create_df_train(df_image, df_study):
    
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
    
    # full columns merged data
    df_size = pd.read_csv('size.csv')
    df_train = df_train.merge(df_size, on='id')
    
    last_df = df_train.drop(["boxes","label","StudyInstanceUID","dim0","dim1","split"], axis = 1)
    
    return df_train, last_df

def save_to_csv(df_train, filename):
    df_train.to_csv(f"{filename}.csv", index = False)
    print("Csv file created...")

df_image, df_study = load_data(image_path, study_path)
_, last_df = create_df_train(df_image, df_study)

filename = "data"
save_to_csv(last_df, filename)
