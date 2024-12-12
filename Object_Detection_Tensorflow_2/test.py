import time
print("1")
time.sleep(1)
print("2",end="\r",flush=True)
time.sleep(1)
print("\033[1A")

