import os
import pickle
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import LSTM
import time
import AverageMeter
import Model as m


def main():
    # Parameters
    RESUMEFROMCHECKPOINT = True;
    start_epoch = 0
    NUM_EPOCHS = 100
    EMBEDDING_DIM = 25
    HIDDEN_DIM = 256
    TARGET_DIM = 2
    LEARNING_RATE = 0.1
    MOMENTUM = 0.3
    DROPOUT = 0.3
    LAYERS = 1
    ISDENSE = False


    directory = 'runs/' + str(LAYERS) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = directory + str(EMBEDDING_DIM) + 'EMBD' + '-' + str(HIDDEN_DIM) + 'HDDN' + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    details_path = path + str(DROPOUT) + 'Drop-' + str(LEARNING_RATE) + 'Learn-' + str(MOMENTUM) + 'Mome/'
    if not os.path.exists(details_path):
        os.mkdir(details_path)

    best_prec1 = 0

    # Load Training Data
    data_path = 'Data/100D/'
    infile = open(data_path+"FinalTestingData.pkl", "rb")
    testingData = pickle.load(infile)
    infile.close()
    # Load Labels
    infile = open(data_path+"FinalTestingLabels.pkl", "rb")
    testingLabels = pickle.load(infile)
    infile.close()

    model_object = m.Model(EMBEDDING_DIM, HIDDEN_DIM, TARGET_DIM,DROPOUT,LEARNING_RATE,MOMENTUM,LAYERS,ISDENSE)

    # If a checkpoint was made, continue from it
    if RESUMEFROMCHECKPOINT:
        RESUMEPATH = details_path + 'model_best.pth.tar'
        if os.path.isfile(RESUMEPATH):
            print("=> loading checkpoint '{}'".format(RESUMEPATH))
            checkpoint = torch.load(RESUMEPATH)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_object.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(RESUMEPATH, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(RESUMEPATH))

    # Loop through epochs
    for epoch in range(start_epoch,NUM_EPOCHS):
        print('Epoch: ' + str(epoch))


        # evaluate on validation set

        [prec1, val_loss, val_accuracy] = model_object.validate(testingData, testingLabels, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_object.model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
        print('Accuracy: ', prec1)
        print('Best accuracy: ', best_prec1)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/' + 'model_best.pth.tar')


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