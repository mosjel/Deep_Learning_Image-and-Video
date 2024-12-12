import glob
j=0
for i in glob.glob(r'fire_dataset\test\*'):
    j=j+1
    print(i)
    if j==4: exit()