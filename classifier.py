"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx,grad_output):
        #print(grad_output)
        dx=grad_output*-1
        #print(dx)
        return dx
#def grad_reverse(x):
#    return GradReverse()(x)   


class QAmodel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(QAmodel, self).__init__()
        self.grad_reverse=GradReverse.apply
       
            
        self.l1=weight_norm(nn.Linear(in_dim, hid_dim), dim=None)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(dropout, inplace=True)
        self.l2=weight_norm(nn.Linear(hid_dim, out_dim), dim=None)

  

    def forward(self, x):
        
        x.register_hook(print)
        x=self.grad_reverse(x)
        x.register_hook(print)
        l1=self.l1(x)
        relu=self.relu(l1)
        drop=self.drop(relu)
        logits=self.l2(drop)

        #logits = self.main(x)
        
        return logits
  