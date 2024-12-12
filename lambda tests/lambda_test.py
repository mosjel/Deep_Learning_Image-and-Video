def myfunc(n):
      
  return lambda a : a * n

s=myfunc(2)
print (s(12))
f=s(12)

output=lambda h:h*f
print (output(10))

numbers=(1,2,3,4)

result=map(lambda x:x*x,numbers)
numbersquare=list(result)
numbersquare1=set(result)
print(numbersquare)