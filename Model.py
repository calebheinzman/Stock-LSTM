import operator
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import LSTM
import time
import AverageMeter

# So far 0.1, 256, 0.4 has yielded the best results after 9 epochs (55% on test)
# 63% on val for 200 glove embedding? Probably not close to accurate. Just got lucky

class Model():

    def __init__(self, embedding_dim, hddn_dim, targ_dim,dropout,lrn_rate,momentum, layers, isDense):
        # Create Model
        self.model = LSTM.LSTM(embedding_dim, hddn_dim, targ_dim,dropout,layers, isDense)
        # Load model to GPU if available
        if (torch.cuda.is_available()):
            self.model.cuda()  # rnn is your model

        # Initializing Loss and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lrn_rate, momentum=momentum)
        self.zero_count = 0
        self.one_count = 0

    def train(self,trainingData, trainingLabels,epoch):
        # Calculates Averages
        batch_time = AverageMeter.AverageMeter()
        losses = AverageMeter.AverageMeter()
        top1 = AverageMeter.AverageMeter()

        # switch to train mode
        self.model.train()

        # Keeps track of time
        end = time.time()

        all_loss = []
        all_accuracy = []

        # Loop through all data
        # print(self.model.lstm.all_weights[0][0][0])
        for i in range(trainingData.__len__()):
            # Extract all messages and labels corresponding to a date
            sentence = trainingData[i]
            target = trainingLabels[i]
            target = target.view(-1).long()

            # Load data to GPU if availible
            if (torch.cuda.is_available()):
                sentence = sentence.cuda()
                target = target.cuda()

            # Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            self.model.zero_grad()

            # Run our forward pass and computer loss
            output = self.model(sentence)


            # The output and target are not aligned.
            # In the example the output consists of 5x3 2d array.
            # Each element in the 5 array corresponds to one word.
            # Each element in the word array corresponds the the classficiation (DET, NN, V)
            # However, the target consists of only a 5 array corresponding to each.
            # The target is suppose to be a value between 0-2 for each word's classifications.

            # So in my case, I think instead of the ouput being a 12 element array, I think
            # it needs to have 12x2 array. Where each 2d array is a prediction on the class 0 or 1.
            loss = self.loss_function(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), 1)
            top1.update(prec1.item(), 1)

            # Compute the gradients, and update the parameters by
            #  calling optimizer.step()
            self.optimizer.zero_grad()
            loss.backward()
            # if i == 290:
            #     print(self.model.lstm.all_weights[0][0][0])
            self.optimizer.step()

            # print(sentence.size())
            _, pred = output.topk(max((1,)), 1, True, True)
            pred = pred.t()
            pred = pred.tolist()
            # print('Target:____ ' + str(target.tolist()))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(trainingData), batch_time=batch_time,
                    loss=losses, top1=top1))

            all_loss.append(losses.val)
            all_accuracy.append(top1.val/100)


        self.oldWeights = self.model.lstm.all_weights[0][0][0]

        return[all_loss,all_accuracy]


        # Set single data point from date.



    def validate(self, validationData, validationLabels, epoch):
        # Calculates Averages
        batch_time = AverageMeter.AverageMeter()
        losses = AverageMeter.AverageMeter()
        top1 = AverageMeter.AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        all_loss = []
        all_accuracy = []

        # confCount = 0
        # correctConfCount = 0
        # unConfCount = 0
        # correctUnconfCount = 0
        # wrongConfCount = 0
        # correctCount = 0
        # wrongCount = 0

        # Keeps track of time
        end = time.time()
        # Loop through all data
        for i in range(validationData.__len__()):
            # Extract all messages and labels corresponding to a date
            sentence = validationData[i]
            target = validationLabels[i]
            target = target.view(-1).long()

            # Load data to GPU if availible
            if (torch.cuda.is_available()):
                sentence = sentence.cuda()
                target = target.cuda()

            # Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            self.model.zero_grad()

            # Run our forward pass.
            output = self.model(sentence)


            # Compute the loss
            _, pred = output.topk(max((1,)), 1, True, True)
            pred = pred.t()
            pred = pred.tolist()
            loss = self.loss_function(output, target)

            # out = output.tolist()[0]
            # prect, out = max(enumerate(out), key=operator.itemgetter(1))
            # tar = target.tolist()[0]
            # conf = 0.8
            # if prect == tar:
            #     if out > conf:
            #         correctConfCount += 1
            #     else:
            #         unConfCount += 1
            #     correctCount +=  1
            # else:
            #     if out > conf:
            #         wrongConfCount += 1
            #     wrongCount += 1



            # Measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), 1)
            top1.update(prec1.item(), 1)

            targetTemp = target.tolist()[0]
            if targetTemp == 0:
                self.zero_count = self.zero_count + 1
            else:
                self.one_count = self.one_count + 1

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(validationData), batch_time=batch_time,
                    loss=losses, top1=top1))

            all_loss.append(losses.val)
            all_accuracy.append(top1.val / 100)

        # ratioConf = (correctConfCount) / (wrongConfCount + correctConfCount)
        # ratio = correctCount / (wrongCount + correctCount)

        return [top1.avg, all_loss, all_accuracy]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)


    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.tolist()
    pred[0] = [int(x) for x in pred[0]]
    pred = torch.tensor(pred)
    pred = pred
    target = target.view(-1).long()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res