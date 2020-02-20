import pickle
import torch

# Load training Data
infile = open("Data/TrainingData.pkl", "rb")
trainingData = pickle.load(infile)
infile.close()
trainingLabels = []

# Load Validation Data
infile = open("Data/ValidationData.pkl", "rb")
validationData = pickle.load(infile)
infile.close()
validationLabels = []

# Loop through training data and seperate daily messages from labels
for i in range(0,trainingData.__len__()):
    day = trainingData[i]
    sentence = []
    for j in range(1,day.__len__()-1):
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

# Loop through validation data and seperate daily messages from labels
for i in range(0,validationData.__len__()):
    day = validationData[i]
    sentence = []
    for j in range(1,day.__len__()-1):
        message = day[j]
        for k in range(0,message.__len__()):
            word = message[k]
            word = [float(x) for x in word]
            sentence.append(word)
    label = day[day.__len__()-1]
    label = [float(x) for x in label]
    validationLabels.append(torch.tensor(label))
    sentence = torch.tensor(sentence)
    sentence = sentence.view(len(sentence), 1, -1)
    validationData[i] = sentence

# Write training data
f = open('FinalTrainingData.pkl','wb')
pickle.dump(trainingData,f)
f.close()
# Write training Labels
f = open('Data/FinalTrainingLabels.pkl','wb')
pickle.dump(trainingLabels,f)
f.close()
# Write Validation Data
f = open('Data/FinalValidationData.pkl','wb')
pickle.dump(validationData,f)
f.close()
# Write Validation Labels
f = open('Data/FinalValidationLabels.pkl','wb')
pickle.dump(validationLabels,f)
f.close()