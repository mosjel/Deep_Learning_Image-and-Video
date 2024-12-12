import sys, time

for i in range(0, 101, 10):
  print("\r>> You have finished {}%".format(i), end='')

#   sys.stdout.flush()
  time.sleep(0.5)
print