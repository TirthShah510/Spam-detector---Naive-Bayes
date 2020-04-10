import matplotlib as plot
import numpy as np
import sys
import re
import os

path = "../src/train/"
dir = os.listdir(path)
dict = {}


for i in dir:

    file = open(path+i)

    l = file.read()
    l = l.lower()
    # print(l)

    words = re.split('[^a-zA-Z]+', l)

    for word in words:
        if word not in dict.keys():
            dict[word] = 1
        else:
            dict[word] += 1

print(dict)