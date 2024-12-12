import pandas as pd
from pathlib import Path
current_path_file=Path(__file__)
address=current_path_file.with_name("Ham_Image_Specifications_detic.feather")

# df=pd.read_feather(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\Detectron2\Detic\feather_file\1_1000.feather")
df1=pd.read_feather(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\RESNET50\Pure_resnet\resnet_features1.feather")
df2=pd.read_feather(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\RESNET50\Pure_resnet\resnet_features2.feather")

# print(df.shape)
print(df1.shape)
print(df2.shape)
final_file=pd.concat([df1,df2],axis=0,ignore_index=True)
print(final_file)
print(final_file.shape)
print(address)
final_file.to_feather(address)




