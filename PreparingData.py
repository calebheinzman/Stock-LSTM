from datetime import datetime
import pickle
import numpy as np


# Load message data
infile = open("Data.pkl", "rb")
messageData = pickle.load(infile)
infile.close()
#Load label Data
infile = open("Labels.pkl", "rb")
labelData = pickle.load(infile)
infile.close()

# # Get rid of first date 11/20 for consistency.
# del messageData[0]

# Loop through all messages and merge them with their labels
for i in range(messageData.__len__()):
    row = messageData[i]
    date = datetime.strptime(row[0], "%m-%d-%y")
    date = datetime.strftime(date, "%Y-%m-%d")
    label = labelData[date]
    if (label.__len__() > 30):
        print(label.__len__())
        print(date)
    messageData[i].append(label)

# Convert to numpy array so that it can be shuffled
messageData = np.array(messageData)
np.random.shuffle(messageData)

# Split training and testing data
dataSize = messageData.size
fullTrainingDataSize = round(dataSize * 0.8)
trainingData = messageData[0:fullTrainingDataSize]
testingData = messageData[fullTrainingDataSize:dataSize]

# Split Training into training and validation data
trainingDataSize = round(.8 * fullTrainingDataSize)
validationData = trainingData[trainingDataSize:fullTrainingDataSize]
trainingData = trainingData[0:trainingDataSize]


# Write Training Data
f = open('TrainingData.pkl','wb')
pickle.dump(trainingData,f)
f.close()

#Write Validation Data
f = open('ValidationData.pkl','wb')
pickle.dump(validationData,f)
f.close()

#Write Testing Data
f = open('TestingData.pkl','wb')
pickle.dump(testingData,f)
f.close()
