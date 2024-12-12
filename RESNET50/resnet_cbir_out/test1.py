
import time,threading

start=time.time()



answer=input('This file already exists, Do you want to overwrite it?')


print(answer)
if answer in ['no', "NO" , "No"]:
    print ('dddd')
elif answer in ['yes', "YES" , "Yes"]:
    print ('fffff')
else:
    print('ffffffmmmmmmmmmmmmmmmmmmm')
    pass
counter=100
day=0
min=0
hours=0


def timecount(elapsed,counter):
    day=0
    min=0
    hours=0
    tottime=elapsed*counter
    if tottime>86400:
        day=tottime//86400
        tottime=tottime%86400
    if tottime>3600:
        hours=tottime//3600
        tottime=tottime%3600
    if tottime>60 :
        min=tottime//60
        tottime=tottime%60

    tottime=round(tottime)
    print ('Time to go is: %02d:%02d:%02d:%02d' % (day,hours, min, tottime))

    import time
  
# define the countdown func.
# def countdown(t):
    
#     while t:
#         mins, secs = divmod(t, 60)
#         timer = '{:02d}:{:02d}'.format(mins, secs)
#         print('\r>>{}'.format(timer),end='')
#         time.sleep(1)
#         t -= 1
#     global linelen
#     linelen=len(timer)
    

    
  
# # function call
end=time.time()
elapsed=end-start
print(elapsed)
timecount(elapsed,10000)
# t1=threading.Thread(countdown(int(elapsed*10000),args=(10,)))
# t2=threading.Thread(print(linelen,args=(10,)))
# t1.start()
# t2.start()

# t1.join()
# t2.join()