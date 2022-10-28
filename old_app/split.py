import pandas as pd
import os



data = pd.read_csv("train_data.csv")
img_dir = "images/train"
    
df_data = data.copy()
drop_df = pd.read_excel("dropped_image_IDs.xlsx") + ".jpg"

drop_index = []
for row in drop_df.values:
    drop_index.append(df_data[df_data["id"] == row[0]].index[0])
    
    
df_data = df_data.drop(drop_index, axis = 0)

files = os.listdir("images/train")

not_in_files_index = []

for file_id in df_data.id:
    if file_id in files:
        continue
    else:
        not_in_files_index.append(df_data[df_data["id"] == file_id].index[0])
        
df_data = df_data.drop(not_in_files_index, axis = 0)