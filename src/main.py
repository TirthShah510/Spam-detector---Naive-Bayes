import matplotlib as plot
import numpy as np
import sys
import math
import re
import os


def wordcount(path):
    dir = os.listdir(path)
    dict = {}
    dict_sorted = {}
    totalwords = 0

    for i in dir:

        file = open(path + i)

        l = file.read()
        l = l.lower().strip()
        # print(l)

        words = re.split(r'[^a-zA-Z]', l)

        for word in words:
            if word.strip() not in dict.keys():
                dict[word] = 1
            else:
                dict[word] += 1



    del dict[""]

    # Added smoothing
    for key in dict.keys():
        dict[key] = dict[key] + 0.5

    for value in dict.values():
        totalwords += value

    for key in sorted(dict.keys()):
        dict_sorted[key] = dict[key]

    return dict_sorted, totalwords


def merge_dictionary(dictspam, dictham):
    final_dict = {}
    temp_dict = {}
    line_no = 1

    # Creating merged dictionary with "word spam ham" format
    for word in dictspam.keys():

        temp = [dictspam[word]]
        temp_dict[word] = temp

        if word in dictham.keys():
            temp_dict[word].append(dictham[word])
            del dictham[word]
        else:
            temp_dict[word].append(0)

    # Adding those words which are only in Ham
    for word in dictham.keys():
        temp_dict[word] = [0]
        temp_dict[word].append(dictham[word])

    # Sorting the dictionary
    for key in sorted(temp_dict.keys()):
        final_dict[key] = temp_dict[key]

    return final_dict


def compute_model(final_dict, spamwords, hamwords):
    model = {}
    vocabulary = len(final_dict)
    file = open("ans.txt", 'w')
    i = 1
    for key in final_dict.keys():
        # print(str(final_dict[key][0] + 1) + "As" + str((final_dict[key][0] + 1) / (spamwords + vocabulary)))
        # print(final_dict[key][1])

        string = str(i) + "  " + key + "  " + str(final_dict[key][1]) + "  " \
                 + str((final_dict[key][1]) / hamwords) \
                 + "  " + str(final_dict[key][0]) + "  " \
                 + str((final_dict[key][0]) / spamwords) + "\n"
        file.write(string)

        # Creating the model(ham,spam)
        temp_list = [(final_dict[key][1] + 1) / (hamwords + vocabulary),
                     (final_dict[key][0] + 1) / (spamwords + vocabulary)]
        model[key] = temp_list
        i += 1
    file.close()

    return model


def training():
    dict_ham, hamwords = wordcount("../src/train/train_ham/")
    dict_spam, spamwords = wordcount("../src/train/train_spam/")

    print("Ham")
    f = open("ham.txt", 'w')
    for i in dict_ham.keys():
        f.write(i + ":" + str(dict_ham[i]) + "\n")
    print("Spam")
    f = open("spam.txt", 'w')
    for i in dict_spam.keys():
        f.write(i + ":" + str(dict_spam[i]) + "\n")

    model = merge_dictionary(dict_spam, dict_ham)
    print("Model")
    f = open("model.txt", 'w')
    for i in model.keys():
        f.write(i + ":" + str(model[i]) + "\n")

    model = compute_model(model, spamwords, hamwords)

    # for word in model.keys():
    #     print(word+":"+str(model[word])+"\n")

    return model


def testing(model):
    dir = os.listdir("../src/test/")
    ansfile = open("Model_ans.txt", "w")
    testing_words = {}

    for i in dir:

        p_ham = math.log((1000 / 1997), 10)
        p_spam = math.log((997 / 1997), 10)
        file = open("../src/test/" + i)
        l = file.read()
        l = l.lower().strip()

        words = re.split(r'[^a-zA-Z]', l)

        # for word in words:
        #     testing_words[word] = 1

        for key in words:

            if key in model.keys():
                p_ham = p_ham + math.log(model[key][0], 10)
                p_spam = p_spam + math.log(model[key][1], 10)

        if p_ham >= p_spam:
            ans = "ham"
        else:
            ans = "spam"


        ansfile.write(str(i) + "  " + str(p_ham) + "  " + str(p_spam) + "  " + ans + "\n")
    print("Testing")


def main():
    model = training()
    testing(model)


main()
