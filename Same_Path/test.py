from pathlib import Path

p = Path(__file__).resolve().parent
new=p / "09121939088"
new.mkdir(exist_ok=True)
new1=new / "jj.txt"
print(new1)


print(p)
# file=open(p,"w")
# file.write("asdeded"),"\n"
# file.close()