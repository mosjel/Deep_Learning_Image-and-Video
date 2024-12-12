import openpyxl as op
import pandas as pd
import numpy as np
s=pd.read_excel("book1.xlsx")
s[["first Number","last Number"]]=s.iloc[:,3].str.split("-",expand=True)
s=s.iloc[:,3:]
s1=pd.DataFrame(s.iloc[:,1:3])
s1.astype(str)
print(s1.iloc[0,0].isdigit())
s1['Subtract']=np.nan
for i in range(s1.shape[0]-1):
      if s1.iloc[i,0].isdigit(): 
         print(i) 
         s1.iloc[i,2]=int(s1.iloc[i,1])-int(s1.iloc[i,0])+1
print(s1.shape)
print(s1.tail(10))

print('-----------------------------------------------------------')
i=0
# print(s.iloc[0,1].isdigit())
# print(s.iloc[0,1])
s["Bool"]=np.nan
s["Bool"]=False
while True:
   
    
    if   (int(s.iloc[i,2])+1)!=int(s.iloc[i+1,1]) :
                    new_1_n=int(s.iloc[i,2])+1
                    new_2_n=int(s.iloc[i+1,1])-1
                    line = pd.DataFrame({"first Number":new_1_n , "last Number":new_2_n}, index=[i+1])
                    s = pd.concat([s.iloc[0:i+1,:], line, s.iloc[i+1:,:]]).reset_index(drop=True)
                    s.loc[i+1,"Bool"]=True
    i=i+1
    if (s.shape[0]-1==i+1):break        
s.insert(1,"first_OLD_N",s1.iloc[:,0])
s.insert(2,"Last_OLD_N",s1.iloc[:,1])
s.insert(3,"Subtract",s1.iloc[:,2])

print(s.tail(10))
s.to_excel("book2.xlsx")

# # #     print(s)

# # #     if int((s.iloc[i,2])+1)!=s.iloc[i+1,1]
    
