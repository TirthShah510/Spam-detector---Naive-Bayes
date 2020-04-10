import matplotlib as plot
import numpy as np
import sys
import re
import os


def ham_wordcount():
    path = "../src/train/train_ham/"
    dir = os.listdir(path)
    dict_ham = {}

    for i in dir:

        file = open(path + i)

        l = file.read()
        l = l.lower()
        # print(l)

        words = re.split('[^a-zA-Z]+', l)

        for word in words:
            if word not in dict_ham.keys():
                dict_ham[word] = 1
            else:
                dict_ham[word] += 1

    return dict_ham

def spam_wordcount():
    path = "../src/train/train_spam/"
    dir = os.listdir(path)
    dict_spam = {}

    for i in dir:

        file = open(path + i)

        l = file.read()
        l = l.lower()
        # print(l)

        words = re.split('[^a-zA-Z]+', l)

        for word in words:
            if word not in dict_spam.keys():
                dict_spam[word] = 1
            else:
                dict_spam[word] += 1

    return dict_spam

def main():

    dict_ham = ham_wordcount()
    dict_spam = spam_wordcount()

    print(dict_ham)
    print(dict_spam)

main()