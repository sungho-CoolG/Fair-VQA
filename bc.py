"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2,.5], k=3):
        super(BCNet, self).__init__()
        
        self.c = 32
        self.k = k
        self.v_dim = v_dim; self.q_dim = q_dim
        self.h_dim = h_dim; self.h_out = h_out
        
        self.v_net = FCNet([v_dim, 2048], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, 2048], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        
        if None == h_out:
            pass

        else:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
            self.h_net = weight_norm(nn.Linear(14, 1), dim=None)
            self.h_net2 = weight_norm(nn.Linear(v_dim,v_dim), dim=None)
    def forward(self, v, q):
        
        # v_ = v.transpose(1,2).unsqueeze(3)
        
        # q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
        # d_ = torch.matmul(v_, q_) # b x h_dim x v 
        # d_=d_.sum(3)
        # logits = d_.transpose(1,2) # b x v x h_dim
        
        #return logits

      
        #import pdb;pdb.set_trace()
        v_ = self.v_net(v).transpose(1,2).unsqueeze(3)
        q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
        d_ = torch.matmul(v_, q_) # b x h_dim x v x q
        logits = self.h_net(d_) # b x h_dim x v x x 1
        logits=logits.squeeze(3).transpose(1,2)
        logits = self.h_net2(logits)
    
        return logits # b x v x h_dim 

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1,2).unsqueeze(2) # b x d x 1 x v
        q_ = self.q_net(q).transpose(1,2).unsqueeze(3) # b x d x q x 1
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_) # b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        if 1 < self.k:
            logits = logits.unsqueeze(1) # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k # sum-pooling
        return logits


if __name__=='__main__':
    net = BCNet(1024,1024,1024,1024).cuda()
    x = torch.Tensor(512,36,1024).cuda()
    y = torch.Tensor(512,14,1024).cuda()
    out = net.forward(x,y)
