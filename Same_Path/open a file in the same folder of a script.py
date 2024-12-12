from pathlib import Path

p = Path(__file__).with_name('hamed.txt')
file=open(p,"w")
file.write("asdeded"),"\n"
file.close()

