# import time
  
# # define the countdown func.
# def countdown(t):
    
#     while t:
#         mins, secs = divmod(t, 60)
#         timer = '{:02d}:{:02d}'.format(mins, secs)
#         print(timer, end="\r")
#         time.sleep(1)
#         t -= 1
      
#     print('Fire in the hole!!')

# t = input("Enter the time in seconds: ")
  
# # function call
# countdown(int(t))

import time
d=''
for x in range (0,5):  
    b = "Loading" + "." * (4-x)
    # print(' '*len(d),end='\r')
    # print("\r>>{}".format(' '*len(d)),end='')
    print ("\r>>jnj{}".format(b),end='')
    d=b
    time.sleep(1)