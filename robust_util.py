from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_set import WEIGHT_DATA_SET
from data import DATA,DATA_small, DATA_action, DATA_partial_logistic, DATA_partial_logistic_deep,DATA_partial_logistic_deep_gpu, DATA_partial_random, DATA_partial_action, DATA_learn_policy, DATA_defined_prob_eval,DATA_partial_logistic_deep_gpu_soften
from torchvision import transforms
import torchvision
import os
import copy
import torch.utils.data as data


import regression_utility as ru
import abstain_utility as au
from scipy import stats
import warnings
import copy
warnings.filterwarnings('ignore')

torch.set_default_tensor_type('torch.DoubleTensor')
mean0 = 0.6
var0 = 1
d = 2

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.model = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.D_in, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.H)),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Linear(self.H, self.D_out)),
            )

    def forward(self, x):

        return self.model(x)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

def my_softmax(x):
    n = np.shape(x)[0]
    max_x, _ = torch.max(x, dim=1)
    max_x = torch.reshape(max_x, (n, 1))
    exp_x = torch.exp(x - max_x)
    p = exp_x / torch.reshape(torch.sum(exp_x, dim=1), (n, 1))
    p[p<10e-8] = 0
    return p


def train(args, model, loss_type, device, train_loader, n_class, optimizer, epoch):
    model.train()

    for batch_idx, (data, target, weight) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        weight = torch.tensor(weight).to(device)
       
        prob = au.my_softmax(torch.einsum('ij,i->ij', output, weight))
        
        
        y_onehot = torch.DoubleTensor(len(data), n_class).to(device)

        y_onehot.zero_()
        y_onehot.scatter_(1, target.reshape(len(data), 1), 1)

        output_last = au.log_gradient.apply(output, torch.tensor(prob), y_onehot)
       
        output_last.backward(torch.ones(output_last.shape).to(device),retain_graph=True)

        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader)))

def train_regression(args, model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0):
     
    model.train()
    lowerB = -1.0/(2*var0)

    grad_yy = torch.empty([0]).to(device)
    grad_yx = torch.empty([d + 1, 0]).to(device)

    lr2 = 1
    lr2 = lr2 * (10 / (10 + np.sqrt(epoch)))

    lr1 = 1
    lr1 = lr1 * (10 / (10 + np.sqrt(epoch)))
    grad_squ_1 = 0.00001
    grad_squ_2 = 0.00001
    for batch_idx, (data, target, weight) in enumerate(train_loader):
        
        data, target, weight = data.to(device), target.to(device), weight.to(device)
      
        optimizer.zero_grad()
        output = model(data)

        meanY, varY = ru.predict_regression(weight, Myy, Myx, output, mean0, var0)
        grad = ru.M_gradient_gpu(output, meanY, varY, target, Myy, Myx)

        grad_squ_1 = grad_squ_1 + grad[0]**2
        grad_squ_2 = grad_squ_2 + grad[1:]**2

        diff = lr1*(grad[0]) + 0.00000*Myy
        grad_yy = torch.cat([grad_yy, grad[0]])

        grad_yx = torch.cat([grad_yx, grad[1:]], 1)
        preM = Myy
        Myy = preM + lr1*(grad[0]/torch.sqrt(grad_squ_1)) + 0.00000*Myy

        while Myy[0][0] < lowerB:
            Myy = Myy + torch.abs(diff)/2

        Myx = Myx + lr2 *(grad[1:]/torch.sqrt(grad_squ_2)) + 0.00000*Myx

        bs = np.shape(output)[0]
        
        
      
        output_last = ru.regression_gradient.apply(output, torch.tensor(Myx[0:-1]), torch.reshape(target, (bs, 1)), torch.reshape(meanY, (bs, 1)))
        output_last.backward(torch.ones(output_last.shape).to(device),retain_graph=True)
        

        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader)))
    # print(np.shape(grad_yy))
    # print(np.shape(grad_yx))
    # print('gradient:', np.linalg.norm(grad_yy))
    # print('gradient:', np.linalg.norm(grad_yx))

    return Myy, Myx,model


def train_MSE(args, model, device, train_loader, optimizer, epoch):
     
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
      
        optimizer.zero_grad()
        output = model(data)

        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        return model
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader)))

def train_weight_MSE(args, model, device, train_loader, optimizer, epoch):
     
    model.train()

    for batch_idx, (data, target, weight) in enumerate(train_loader):
        
        data, target, weight = data.to(device), target.to(device), weight.to(device)
      
        optimizer.zero_grad()
        output = model(data)

        criterion = nn.MSELoss(reduce=False)
        loss = (1 - weight) * criterion(output, target)/(weight * weight)
        loss = torch.mean(loss)
        loss.backward()

        optimizer.step()
    return model
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader)))

def test(args, model, loss_type, device, test_loader, n_class):
    model.eval()
    test_loss = 0
    correct = 0
    prediction = np.empty([0,1])
    probability = np.empty([0,n_class])
    num_acc = 0
    with torch.no_grad():
        for data, target, weight in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            weight = weight.to(device)
            weight_output = torch.einsum('ij,i->ij', (output, weight))
                
            prob = au.my_softmax(weight_output).cpu().numpy()
            probability = np.concatenate((probability, prob))

            criterion = nn.CrossEntropyLoss()
            # loss = criterion(weight_output.clone().detach(), target)
            # test_loss += loss.item()# sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            num_acc += len(data)
          
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred = pred.cpu().numpy()
            if np.shape(pred)[0] !=0 :
                prediction = np.concatenate((prediction, pred)) 
            
            
    test_loss = 1.0 - float(correct)/len(test_loader.dataset)
    # if num_acc !=0:
    #
    #     print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Abstaining rate: {:.0f}%\n'.format(
    #         test_loss, correct, num_acc,
    #         100. * correct / num_acc, 100. *(1- float(num_acc)/len(test_loader.dataset)) ))
    # else:
    #     print('Abstaining rate is 1.')
    return probability, prediction, 100. * correct / len(test_loader.dataset), test_loss


def test_regression(args, model, Myy, Myx, device, test_loader, mean0, var0):
    model.eval()
    test_loss = 0
    y_prediction = torch.empty([1, 0]).to(device)
    y_var = torch.empty([1, 0]).to(device)
    with torch.no_grad():
        for data, target, weight in test_loader:
            data, target = data.to(device), target.to(device)
            weight = weight.to(device)
            output = model(data)
            
            d = np.shape(data)[0]
            target = torch.reshape(target, (1, d))
            
            meanY, varY = ru.predict_regression(weight, Myy, Myx, output, mean0, var0)
            criterion = nn.MSELoss()
            l2loss = criterion(meanY, target)
            test_loss += torch.sum(l2loss)
            y_prediction = torch.cat([y_prediction, meanY], axis=1)

            y_var = torch.cat([y_var, varY], axis = 1)
            

    test_loss /= len(test_loader.dataset)
    return y_prediction, y_var, test_loss

def test_MSE(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    y_prediction = np.empty([1, 0])
  
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
        
            criterion = nn.MSELoss()
            l2loss = criterion(output, target)
            test_loss += torch.sum(l2loss)
            d = np.shape(data)[0]
            meanY = np.reshape(output.cpu(), (1,d))

            y_prediction = np.concatenate((y_prediction, meanY), axis=1)

        
    test_loss /= len(test_loader.dataset)
    # print('Average loss: {:.4f}\n'.format(test_loss))
    # print(target)
    # print(meanY)
    return y_prediction, test_loss


def test_weight_MSE(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    y_prediction = np.empty([1, 0])
  
    with torch.no_grad():
        for data, target, weight, in test_loader:
            data, target, weight = data.to(device), target.to(device), weight.to(device)
            
            output = model(data)
            
            criterion = nn.MSELoss(reduce = False)
            l2loss = (1 - weight)* criterion(output, target)/(weight*weight)
            l2loss = torch.mean(l2loss)
            test_loss += torch.sum(l2loss)
            d = np.shape(data)[0]
            meanY = np.reshape(output.cpu(), (1,d))

            y_prediction = np.concatenate((y_prediction, meanY), axis=1)

        
    test_loss /= len(test_loader.dataset)
    # print('Average loss: {:.4f}\n'.format(test_loss))
    # print(target)
    # print(meanY)
    return y_prediction, test_loss

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_validate_test(args, lr,epoch, loss_type, device, use_cuda, train_model, train_loader, test_loader, validate_loader, n_class, lbd, testflag = True):
    if loss_type == 'logloss':      
        optimizer = optim.Adam(train_model.parameters(), lr=lr, weight_decay=lbd)
        for epoch in range(1, epoch + 1):
            train(args, train_model, loss_type, device, train_loader, n_class, optimizer, epoch) 
            probability, predictions, acc, loss = test(args, train_model,loss_type, device, validate_loader, n_class)
        
        # print('\nTesting on test set')
        
        if testflag == True:
        
            probability, predictions, acc, loss = test(args, train_model,loss_type, device, test_loader, n_class)
        
        return probability, predictions, acc, loss
        # f1 = f1_score(test_labels, predictions, average='macro') 
        # acc_per_class = acc_perclass(test_labels, predictions, n_class)
        # print('F1-score:', f1)
        # print('Per class accuracy', acc_per_class)
    elif loss_type == 'regression':
        inbest = 0
        now_best = 0
        best_loss = 100000000
        best_epoch = 1
        Myy = torch.ones((1, 1)).to(device)
        Myx = torch.ones((d+1, 1)).to(device)
        best_Myy = torch.ones((1, 1)).to(device)
        best_Myx = torch.ones((1, 1)).to(device)
        best_model = copy.deepcopy(train_model)
        optimizer = optim.SGD(train_model.parameters(), lr=lr, momentum=args.momentum, weight_decay=lbd)
        for epoch in range(1, epoch + 1):
            Myy, Myx,train_model = train_regression(args, train_model, device, train_loader, optimizer, epoch, Myy, Myx, mean0, var0) 
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, validate_loader, mean0, var0)
            # if loss < best_loss:
            #     best_epoch = epoch
            #     best_loss = loss
            #     best_model  = copy.deepcopy(train_model)
            #     best_Myx =  copy.deepcopy(Myx)
            #     best_Myy =  copy.deepcopy(Myy)
            #     inbest = 1
        #     if epoch - best_epoch>20:
        #         print(epoch)
        #         if inbest == 1:
        #             train_model = best_model
        #             Myy = best_Myy
        #             Myx = best_Myx
        #             now_best = 1
        #         break
        # # print('\nTesting on test set')
        # if now_best == 0:
        #     train_model = best_model
        #     Myy = best_Myy
        #     Myx = best_Myx
        if testflag == True:
            meanY, varY, loss = test_regression(args, train_model, Myy, Myx, device, test_loader, mean0, var0)
       
        return train_model, Myy, Myx, meanY, varY, loss
    elif loss_type == 'mse':
        optimizer = optim.Adam(train_model.parameters(), lr=lr, weight_decay=lbd)
        for epoch in range(1, epoch + 1):
            best_model = copy.deepcopy(train_model)
            train_model = train_MSE(args, train_model, device, train_loader, optimizer, epoch) 
            pred_Y, loss = test_MSE(args, train_model,  device, validate_loader)
        return train_model, pred_Y, loss


def round_value(a):
    if a>1:
        a = 1.0
    elif a<0:
        a = 0
    return a



def my_bound(x):
    if isinstance(x, np.ndarray):  
        x[x>1000] = 1000.0
        x[x<0.001] = 0.001
        x[np.isnan(x)] = 0.001
    elif torch.is_tensor(x):
        x = x.cpu().numpy()
        x[x>1000] = 1000.0
        x[x<0.001] = 0.001
        x[np.isnan(x)] = 0.001
    else:
        if x>1000:
            x = 1000.0
        elif x<0.001:
            x = 0.001
    return x

