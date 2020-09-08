import numpy as np
import math
import re
import os
import matplotlib.pyplot as plt


def wordCount(path):
    dir = os.listdir(path)
    dictHam = {}
    dictSpam = {}
    sortedHamDict = {}
    sortedSpamDict = {}
    totalHamWords = 0
    totalSpamWords = 0

    for i in dir:
        if "ham" in i:

            fileHam = open(path + i, encoding="ISO-8859-1")

            lHam = fileHam.read()
            lHam = lHam.lower().strip()

            wordsHam = re.split(r'[^a-zA-Z]', lHam)

            for word in wordsHam:
                if word.strip() not in dictHam.keys():
                    dictHam[word] = 1
                else:
                    dictHam[word] += 1
        else:

            fileSpam = open(path + i, encoding="ISO-8859-1")

            lSpam = fileSpam.read()
            lSpam = lSpam.lower().strip()

            wordsSpam = re.split(r'[^a-zA-Z]', lSpam)

            for word in wordsSpam:
                if word.strip() not in dictSpam.keys():
                    dictSpam[word] = 1
                else:
                    dictSpam[word] += 1

    del dictHam[""]
    del dictSpam[""]

    for value in dictHam.values():
        totalHamWords += value

    for value in dictSpam.values():
        totalSpamWords += value

    for key in sorted(dictHam.keys()):
        sortedHamDict[key] = dictHam[key]

    for key in sorted(dictSpam.keys()):
        sortedSpamDict[key] = dictSpam[key]

    fileHam.close()
    fileSpam.close()
    return sortedHamDict, totalHamWords, sortedSpamDict, totalSpamWords


def mergeDictionary(spamDict, hamDict):
    finalDict = {}
    tempDict = {}

    # Creating merged dictionary with "word spam ham" format
    for word in spamDict.keys():

        temp = [spamDict[word]]
        tempDict[word] = temp

        if word in hamDict.keys():
            tempDict[word].append(hamDict[word])
            del hamDict[word]
        else:
            tempDict[word].append(0)

    # Adding those words which are only in Ham
    for word in hamDict.keys():
        tempDict[word] = [0]
        tempDict[word].append(hamDict[word])

    # Sorting the dictionary
    for key in sorted(tempDict.keys()):
        finalDict[key] = tempDict[key]

    return finalDict


def computeModel(finalDict, spamWords, hamWords):
    model = {}
    vocabulary = len(finalDict)

    # Format of File-> Same as given in project description
    modelFile = open("Model.txt", 'w', encoding="ISO-8859-1")
    i = 1
    for key in finalDict.keys():

        # Creating the model(ham,spam)
        # Smoothing added (0.5)
        tempList = [(finalDict[key][1] + 0.5) / (hamWords + (vocabulary * 0.5)),
                    (finalDict[key][0] + 0.5) / (spamWords + (vocabulary * 0.5))]

        model[key] = tempList

        string = str(i) + "  " + key + "  " + str(finalDict[key][1]) + "  " + str(tempList[0]) + "  " \
                 + str(finalDict[key][0]) + "  " + str(tempList[1]) + "\n"
        modelFile.write(string)

        i += 1

    modelFile.close()
    return model


def training():
    hamDict, hamWords, spamDict, spamWords = wordCount("../src/train/")
    #spamDict, spamWords = wordCount("../src/train/train_spam/")

    hamWordFrequencyFile = open("HamWordFrequency.txt", 'w', encoding="ISO-8859-1")
    for i in hamDict.keys():
        hamWordFrequencyFile.write(i + ":" + str(hamDict[i]) + "\n")
    spamWordFrequncyFile = open("SpamWordFrequency.txt", 'w')
    for i in spamDict.keys():
        spamWordFrequncyFile.write(i + ":" + str(spamDict[i]) + "\n")

    model = mergeDictionary(spamDict, hamDict)
    hamAndSpamWordFrequencyFile = open("HamAndSpamWordFrequency.txt", 'w')
    for i in model.keys():
        # Formate of File->  Word:[Frequency Count in Spam, Frequency Count in Ham]
        hamAndSpamWordFrequencyFile.write(i + ":" + str(model[i]) + "\n")

    model = computeModel(model, spamWords, hamWords)
    print("Model Training Completed")

    hamWordFrequencyFile.close()
    spamWordFrequncyFile.close()
    hamAndSpamWordFrequencyFile.close()
    return model


def testing(model):

    dir = os.listdir("../src/test/")

    # Formate for File-> Same as given in project description
    resultFile = open("Result.txt", "w")
    lineCounter = 1

    spamHam = 0  # count of Spam misclassified into  Ham
    hamSpam = 0  # count of Ham misclassified into  Spam
    spamSpam = 0  # count of Spam correctly classified into Spam
    hamHam = 0  # count of Ham correctly Classified into Ham
    
    for i in dir:
        if "ham" in i:
            fileType = "ham"
        else:
            fileType = "spam"

        probabilityOfHam = math.log((1000 / 1997), 10)
        probabilityOfSpam = math.log((997 / 1997), 10)
        file = open("../src/test/" + i, encoding="ISO-8859-1")
        l = file.read()
        l = l.lower().strip()

        words = re.split(r'[^a-zA-Z]', l)


        for key in words:
            if key in model.keys():
                probabilityOfHam = probabilityOfHam + math.log(model[key][0], 10)
                probabilityOfSpam = probabilityOfSpam + math.log(model[key][1], 10)

        resultLabel = "wrong"
        if probabilityOfHam >= probabilityOfSpam:
            ans = "ham"
            if fileType is ans:
                hamHam += 1
                resultLabel = "right"
            else:
                spamHam += 1
        else:
            ans = "spam"
            
            if fileType is "spam":
                spamSpam += 1
                resultLabel = "right"
            else:
                hamSpam += 1

        resultFile.write(str(lineCounter) + "  " + str(i) + "  " + ans + "  " + str(probabilityOfHam) + "  " + str(probabilityOfSpam) + "  " +
                      fileType + "  " + resultLabel + "\n")
        lineCounter = lineCounter + 1

    resultFile.close()
    print("Model testing Completed")

    precision = hamHam / (hamHam + spamHam)
    recall = hamHam / (hamHam + hamSpam)
    accuracy = (hamHam + spamSpam) / (hamHam + spamSpam + spamHam + hamSpam)
    f1_measure = (2 * precision * recall) / (precision + recall)

    print("-----------------------------------------------------------------------------------------------------------")
    print("HAM CLASS:\n ")
    print("TP = " + str(hamHam) + "  FP = " + str(spamHam) + "  FN = " + str(hamSpam) + "  TN = " + str(spamSpam))
    print("HAM CLASS : Confusion Matrix")
    print("[[" + str(hamHam) + " " + str(spamHam) + "]")
    print(" [ " + str(hamSpam) + " " + str(spamSpam) + " ]]")
    print("Precision = " + str(precision) + "  Recall = " + str(recall) + "  Accuracy = " + str(accuracy) +
          "  F1-Measure = " + str(f1_measure))

    precision = spamSpam / (spamSpam + hamSpam)
    recall = spamSpam / (spamSpam + spamHam)
    accuracy = (hamHam + spamSpam) / (hamHam + spamSpam + spamHam + hamSpam)
    f1_measure = (2 * precision * recall) / (precision + recall)

    print("-----------------------------------------------------------------------------------------------------------")
    print("SPAM CLASS:\n ")
    print("TP = " + str(spamSpam) + "  FP = " + str(hamSpam) + "  FN = " + str(spamHam) + "  TN = " + str(hamHam))
    print("SPAM CLASS : Confusion Matrix")
    print("[[" + str(spamSpam) + " " + str(hamSpam) + "]")
    print(" [ " + str(spamHam) + " " + str(hamHam) + " ]]")
    print("Precision = " + str(precision) + "  Recall = " + str(recall) + "  Accuracy = " + str(accuracy) +
          "  F1-Measure = " + str(f1_measure))

    print("\n---------------------------------------------------------------------------------------------------------")

    arrayForHamClass = np.array([[hamHam, spamHam], [hamSpam, spamSpam]])
    plotConfusionMatrix(arrayForHamClass, "ham")
    arrayForSpamClass = np.array([[spamSpam, hamSpam], [spamHam, hamHam]])
    plotConfusionMatrix(arrayForSpamClass, "spam")


def plotConfusionMatrix(array, category):
    configArray = array
    normConfig = []
    for i in configArray:
        tempArray = []
        a = sum(i, 0)
        for j in i:
            tempArray.append(float(j) / float(a))
        normConfig.append(tempArray)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(normConfig), cmap=plt.cm.jet,
                    interpolation='nearest')
    width, height = configArray.shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(configArray[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.xticks(np.arange(width), (category, "not " + category))
    plt.yticks(np.arange(height), (category, "not " + category))
    plt.xlabel("Actual Output")
    plt.ylabel("Model Output")
    plt.savefig('ConfusionMatrix-' + category + '.png', format='png')

    print("Confusion Matrix File for " + category + " class created.")


def main():
    print("-----------------------------------------------------------------------------------------------------------")
    print("Model Training Started")
    model = training()
    print("-----------------------------------------------------------------------------------------------------------")
    print("Model Testing Started")
    testing(model)


main()
