import os
import pickle
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import LSTM
import time
import AverageMeter
import Model as m
import matplotlib.pyplot as plt


def main():
    #128,256,32,64 Already Done
    hdn_dims = [128,256,32,64,512]
    for dim in hdn_dims:
        print("Dimension: " + str(dim))
        # Parameters
        RESUMEFROMCHECKPOINT = False;
        start_epoch = 0
        NUM_EPOCHS = 50
        EMBEDDING_DIM = 25
        HIDDEN_DIM = dim
        TARGET_DIM = 2
        LEARNING_RATE = 0.1
        MOMENTUM = 0.3
        DROPOUT = 0.3
        LAYERS = 1
        ISDENSE = False

        if ISDENSE:
            directory = 'runs/Dense' + str(LAYERS) + '/'
        else:
            directory = 'runs/'+str(LAYERS) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = directory + str(EMBEDDING_DIM) + 'EMBD' + '-' + str(HIDDEN_DIM)+ 'HDDN' + '/'
        if not os.path.exists(path):
            os.mkdir(path)
        details_path = path + str(DROPOUT) + 'Drop-' + str(LEARNING_RATE) + 'Learn-' + str(MOMENTUM) + 'Mome/'
        if not os.path.exists(details_path):
            os.mkdir(details_path)


        best_prec1 = 0

        # Load Training Data
        data_path = 'Data/100D/'
        infile = open(data_path+"FinalTrainingData.pkl", "rb")
        trainingData = pickle.load(infile)
        infile.close()
        # Load Labels
        infile = open(data_path+"FinalTrainingLabels.pkl", "rb")
        trainingLabels = pickle.load(infile)
        infile.close()
        # Load Training Data
        infile = open(data_path+"FinalValidationData.pkl", "rb")
        validationData = pickle.load(infile)
        infile.close()
        # Load Labels
        infile = open(data_path+"FinalValidationLabels.pkl", "rb")
        validationLabels = pickle.load(infile)
        infile.close()


        model_object = m.Model(EMBEDDING_DIM, HIDDEN_DIM, TARGET_DIM,DROPOUT,LEARNING_RATE,MOMENTUM,LAYERS,ISDENSE)

        # If a checkpoint was made, continue from it
        if RESUMEFROMCHECKPOINT:
            RESUMEPATH = details_path + 'checkpoint.pth.tar'
            if os.path.isfile(RESUMEPATH):
                print("=> loading checkpoint '{}'".format(RESUMEPATH))
                checkpoint = torch.load(RESUMEPATH)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model_object.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(RESUMEPATH, checkpoint['epoch']))

                # Load all previous losses and accuracies
                infile = open(details_path+'loss.pkl', "rb")
                all_loss = pickle.load(infile)
                infile.close()
                infile = open(details_path+'accuracy.pkl', "rb")
                all_accuracy = pickle.load(infile)
                infile.close()
                infile = open(details_path+'avg_loss.pkl', "rb")
                average_loss = pickle.load(infile)
                infile.close()
                infile = open(details_path+'avg_accuracy.pkl', "rb")
                average_accuracy = pickle.load(infile)
                infile.close()
                # Load all val loss and accuracies
                infile = open(details_path+'val_loss.pkl', "rb")
                all_val_loss = pickle.load(infile)
                infile.close()
                infile = open(details_path+'val_accuracy.pkl', "rb")
                all_val_accuracy = pickle.load(infile)
                infile.close()
                infile = open(details_path+'avg_val_loss.pkl', "rb")
                average_val_loss = pickle.load(infile)
                infile.close()
                infile = open(details_path+'avg_val_accuracy.pkl', "rb")
                average_val_accuracy = pickle.load(infile)
                infile.close()

            else:
                print("=> no checkpoint found at '{}'".format(RESUMEPATH))
                # Initialize everything to keep track of loss / accuracies
                all_loss = []
                all_accuracy = []
                average_loss = []
                average_accuracy = []
                all_val_loss = []
                all_val_accuracy = []
                average_val_loss = []
                average_val_accuracy = []
        else:
            # Initialize everything to keep track of loss / accuracies
            all_loss = []
            all_accuracy = []
            average_loss = []
            average_accuracy = []
            all_val_loss = []
            all_val_accuracy = []
            average_val_loss = []
            average_val_accuracy = []

        # Loop through epochs
        for epoch in range(start_epoch,NUM_EPOCHS):
            print('Epoch: ' + str(epoch))

            [epoch_loss,epoch_accuracy] = model_object.train(trainingData, trainingLabels,epoch)

            #Keep Track of all the Loss and accuracy info
            all_loss = all_loss + epoch_loss
            all_accuracy = all_accuracy + epoch_accuracy
            avg_loss = sum(epoch_loss)/len(epoch_loss)
            avg_accuracy = sum(epoch_accuracy)/len(epoch_accuracy)
            average_loss.append(avg_loss)
            average_accuracy.append(avg_accuracy)

            # evaluate on validation set
            [prec1, val_loss, val_accuracy] = model_object.validate(validationData, validationLabels, epoch)

            # Keep track of all the val loss and accuracy info
            all_val_loss = all_val_loss + val_loss
            all_val_accuracy = all_val_accuracy + val_accuracy
            avg_val_loss = sum(val_loss)/len(val_loss)
            avg_val_accuracy = sum(val_accuracy)/len(val_accuracy)
            average_val_loss.append(avg_val_loss)
            average_val_accuracy.append(avg_val_accuracy)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_object.model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best,best_prec1,details_path,average_loss,average_accuracy,all_loss,all_accuracy,average_val_loss,average_val_accuracy,all_val_loss,all_val_accuracy)
            print('Accuracy: ', prec1)
            print('Best accuracy: ', best_prec1)

def save_checkpoint(state, is_best,best_prec1,path,average_loss,average_accuracy,all_loss,all_accuracy,average_val_loss,average_val_accuracy,all_val_loss,all_val_accuracy, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""

    filename = path + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, path + 'model_best.pth.tar')

    # Plot the loss and accuracy
    plt.plot(average_loss, 'r', average_val_loss, 'g')
    plt.plot(average_accuracy, 'r', average_val_accuracy, 'g',linestyle = 'dashed')
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epoch')
    plt.title('Loss/Accuracy Over Time')
    locs, labels = plt.xticks()
    plt.xticks(np.arange(0, average_accuracy.__len__(), step=1))
    plt.savefig(path+'loss-accr.png')
    plt.clf()

    # Write Average Training Loss
    f = open(path+'avg_loss.pkl', 'wb')
    pickle.dump(average_loss, f)
    f.close()
    # Write Average Training Accuracy
    f = open(path+'avg_accuracy.pkl', 'wb')
    pickle.dump(average_accuracy, f)
    f.close()
    # Write All Training Loss
    f = open(path+'loss.pkl', 'wb')
    pickle.dump(all_loss, f)
    f.close()
    # Write All Training Accuracy
    f = open(path+'accuracy.pkl', 'wb')
    pickle.dump(all_accuracy, f)
    f.close()

    # Write Average Validation Loss
    f = open(path+'avg_val_loss.pkl', 'wb')
    pickle.dump(average_val_loss, f)
    f.close()
    # Write Average Validation Accuracy
    f = open(path+'avg_val_accuracy.pkl', 'wb')
    pickle.dump(average_val_accuracy, f)
    f.close()
    # Write All Validation Loss
    f = open(path+'val_loss.pkl', 'wb')
    pickle.dump(all_val_loss, f)
    f.close()
    # Write All Labels
    f = open(path+'val_accuracy.pkl', 'wb')
    pickle.dump(all_val_accuracy, f)
    f.close()

    file1 = open(path + 'best.txt', "w")
    file1.write(str(best_prec1))
    file1.close()


if __name__ == '__main__':
    main()



# # See what the scores are after training
# with torch.no_grad():
#     inputs = trainingData[0]
#     tag_scores = model(inputs)
#
#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(tag_scores)
#
# torch.save(model.state_dict(), 'model')