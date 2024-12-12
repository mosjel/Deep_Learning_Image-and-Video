import pandas as pd
df=pd.DataFrame({"a":[1,2,3,5,7],"b":[3,4,5,9,7],"c":[6,7,8,9,7],"4":[3,7,1,1,2]})
print(df)
print(df.shape)
df1=(df.iloc[:,2:])
print(df1.shape)

print(df1)
