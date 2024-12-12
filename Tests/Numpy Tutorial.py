
import numpy as np

a=np.array([1,2,3])
b=np.array([[[1,2],[3,4],[5,6]],[[3,5],[7,8],[9,6]],[[6,7],[8,9],[7,7]]])
c=np.ones((3,2))

#np.save ('8.npy',a)
#np.savez("multiarray.npz",a,c)

#vv=np.load('8.npy')
#print(vv)
d=np.load("multiarray.npz")
print(d['arr_0'])
print(d['arr_1'])

s=np.loadtxt('sm.txt')
print(s)

ss=np.ones((3,4,5))
print(len(ss))

print(c.dtype)
dd=c.astype(np.int32)
print(dd.dtype)


import numpy
txt1=np.loadtxt("sm.txt")
print(txt1)

print(c)
d=c.astype(np.int32)
print(a)
print(b)
jj1=np.array([[1,2,3],
     [4,5,6] ,
  
     [7,8,9]   ])
print (jj1)
print(  np.sum(jj1,axis=1))
np.savez("multiarray.npz",a,b,c,jj1)
hames=np.load("multiarray.npz")
print("-----------------------")
print(hames["arr_3"])