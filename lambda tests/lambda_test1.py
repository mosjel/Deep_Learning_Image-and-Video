def myfunc(n):
    v=n*3
    return lambda a : a * v

s=myfunc(2)
print (s(12))
f=s(12)

output=lambda h:h*f
print (output(10))

numbers=(1,2,3,4)

result=map(lambda x:x*x,numbers)
numbersquare=list(result)
numbersquare1=set(numbersquare)
print(numbersquare)
print(numbersquare1)