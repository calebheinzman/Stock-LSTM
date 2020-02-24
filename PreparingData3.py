# Load message data
import os
import pickle
from datetime import datetime
import random

import numpy as np
import torch

infile = open("Data/100D/DataDictionary.pkl", "rb")
messageData = pickle.load(infile)
infile.close()
#Load label Data
infile = open("Data/100D/LabelsDictionary.pkl", "rb")
labelData = pickle.load(infile)
infile.close()

messages = []
messagesZero = []
messagesOne = []
count = 1
for keyDate in messageData:
    for keySymb in messageData[keyDate]:
        arr = messageData[keyDate][keySymb]
        date = datetime.strptime(keyDate, "%m-%d-%y")
        date = datetime.strftime(date, "%Y-%m-%d")
        label = labelData[date][keySymb]
        arr.append(label)
        if label[0] == 0:
            messagesZero.append(arr)
        else:
            messagesOne.append(arr)
        count = count + 1

classCount = min(messagesOne.__len__(),messagesZero.__len__())
messagesOne = messagesOne[0:classCount]
messagesZero = messagesZero[0:classCount]

# Convert to numpy array so that it can be shuffled
messageDataOne = np.array(messagesOne)
np.random.shuffle(messageDataOne)

# Split training and testing data
dataSizeOne = messageDataOne.size
fullTrainingDataSizeOne = round(dataSizeOne * 0.8)
trainingDataOne = messageDataOne[0:fullTrainingDataSizeOne]
testingDataOne = messageDataOne[fullTrainingDataSizeOne:dataSizeOne]

# Split Training into training and validation data
trainingDataSizeOne = round(.8 * fullTrainingDataSizeOne)
validationDataOne = trainingDataOne[trainingDataSizeOne:fullTrainingDataSizeOne]
trainingDataOne = trainingDataOne[0:trainingDataSizeOne]

# Convert to numpy array so that it can be shuffled
messageDataZero = np.array(messagesZero)
np.random.shuffle(messageDataZero)

# Split training and testing data
dataSizeZero = messageDataZero.size
fullTrainingDataSizeZero = round(dataSizeZero * 0.8)
trainingDataZero = messageDataZero[0:fullTrainingDataSizeZero]
testingDataZero = messageDataZero[fullTrainingDataSizeZero:dataSizeZero]

# Split Training into training and validation data
trainingDataSizeZero = round(.8 * fullTrainingDataSizeZero)
validationDataZero = trainingDataZero[trainingDataSizeZero:fullTrainingDataSizeZero]
trainingDataZero = trainingDataZero[0:trainingDataSizeZero]

trainingData = np.concatenate([trainingDataOne,trainingDataZero])
validationData = np.concatenate([validationDataOne,validationDataZero])
testingData = np.concatenate([testingDataOne,testingDataZero])


np.random.shuffle(trainingData)
np.random.shuffle(validationData)
np.random.shuffle(testingData)

trainingLabels = []
# Loop through training data and seperate daily messages from labels
for i in range(0,trainingData.__len__()):
    print(str(i) + "/" + str(trainingData.__len__()))
    day = trainingData[i]
    sentence = []
    for j in range(0,day.__len__()-1):
        message = day[j]
        for k in range(0,message.__len__()):
            word = message[k]
            word = [float(x) for x in word]
            sentence.append(word)
    label = day[day.__len__()-1]
    # label = [max(float(0),float(x-0.01)) for x in label]
    label = [float(x) for x in label]
    trainingLabels.append(torch.tensor(label))
    sentence = torch.tensor(sentence)
    sentence = sentence.view(len(sentence), 1, -1)
    trainingData[i] = sentence

valLabels = []
# Loop through valdiation data and seperate daily messages from labels
for i in range(0,validationData.__len__()):
    print(str(i) + "/" + str(validationData.__len__()))
    day = validationData[i]
    sentence = []
    for j in range(0,day.__len__()-1):
        message = day[j]
        for k in range(0,message.__len__()):
            word = message[k]
            word = [float(x) for x in word]
            sentence.append(word)
    label = day[day.__len__()-1]
    # label = [max(float(0),float(x-0.01)) for x in label]
    label = [float(x) for x in label]
    valLabels.append(torch.tensor(label))
    sentence = torch.tensor(sentence)
    sentence = sentence.view(len(sentence), 1, -1)
    validationData[i] = sentence

testingLabels = []
# Loop through training data and seperate daily messages from labels
for i in range(0,testingData.__len__()):
    print(str(i) + "/" + str(testingData.__len__()))
    day = testingData[i]
    sentence = []
    for j in range(0,day.__len__()-1):
        message = day[j]
        for k in range(0,message.__len__()):
            word = message[k]
            word = [float(x) for x in word]
            sentence.append(word)
    label = day[day.__len__()-1]
    # label = [max(float(0),float(x-0.01)) for x in label]
    label = [float(x) for x in label]
    testingLabels.append(torch.tensor(label))
    sentence = torch.tensor(sentence)
    sentence = sentence.view(len(sentence), 1, -1)
    testingData[i] = sentence

directory = 'Data/100D/'
if not os.path.exists(directory):
    os.makedirs(directory)

# Write training data
f = open(directory+'FinalTrainingData.pkl','wb')
pickle.dump(trainingData,f)
f.close()
# Write training Labels
f = open(directory+'FinalTrainingLabels.pkl','wb')
pickle.dump(trainingLabels,f)
f.close()
# Write Validation Data
f = open(directory+'FinalValidationData.pkl','wb')
pickle.dump(validationData,f)
f.close()
# Write Validation Labels
f = open(directory+'FinalValidationLabels.pkl','wb')
pickle.dump(valLabels,f)
f.close()
# Write Validation Data
f = open(directory+'FinalTestingData.pkl','wb')
pickle.dump(testingData,f)
f.close()
# Write Validation Labels
f = open(directory+'FinalTestingLabels.pkl','wb')
pickle.dump(testingLabels,f)
f.close()