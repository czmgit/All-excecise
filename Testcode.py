import numpy as np
from functools import reduce
import re
temp = [5,2,3,4,1111,0.2]
line = "john tracelled to the hallway?"

#nid ,str1,line = line.split(" ",2)
list2 = []
for x in re.split('(\W+)?',line):
    if x.strip():
        list2.append(x.strip())
s =set(list2)
dic = sorted(reduce(lambda x,y:x | y,(set(list2))))
print("dic",dic)
print(list2)
print(s)