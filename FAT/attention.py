"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet
from bc import BCNet
from bc_ques import BCNet_q

from outerproduct import outerproduct


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        #w=w*2000
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class CAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(CAttention, self).__init__()

        self.v_proj = FCNet([v_dim, v_dim])
        self.q_proj = FCNet([q_dim, v_dim])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(v_dim,v_dim), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        w=w*2048
        return w

    def logits(self, v, q):
        batch, _ = v.size()
        v_proj = self.v_proj(v) # [batch, qdim]
        q_proj = self.q_proj(q)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits    

class v_on_CAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(v_on_CAttention, self).__init__()

        self.v_proj = FCNet([v_dim, v_dim])
        self.q_proj = FCNet([q_dim, v_dim])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(v_dim,v_dim), dim=None)

    def forward(self, v, q,v_att):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        
        v=v*v_att*v.size(1)
        logits = self.logits(v, q)
        logits=logits*v_att
        logits=logits.sum(1)
        #import pdb;pdb.set_trace()
        w = nn.functional.softmax(logits, 1)
        w=w*2048
        return w

    def logits(self, v, q):
        batch,k,_ = v.size()
        v_proj = self.v_proj(v) # [batch, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits     

class VCAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(VCAttention, self).__init__()

        self.v_proj = FCNet([v_dim, v_dim])
        self.q_proj = FCNet([q_dim, v_dim])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(v_dim,v_dim), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_num=v.size(1)
        logits = self.logits(v, q)
        
        w = nn.functional.softmax(logits.view(-1,v_num*2048), 1)
        w=w*2048
        return w.view(-1,v_num,2048)

    def logits(self, v, q):
        batch,k,_ = v.size()
        v_proj = self.v_proj(v) # [batch, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits    


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p = self.forward_all(v, q, v_mask)
        return p

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q
        logits=logits
        #import pdb;pdb.set_trace()
        '''
        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(2).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))
        '''
        logits=logits.contiguous()
        
        p = nn.functional.softmax(logits.view(-1, v_num * 2048), 1)
        p=p*2048
        #import pdb;pdb.set_trace()
        return p.view(-1, v_num, 2048)

class BiAttention_both(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention_both, self).__init__()

        self.glimpse = glimpse
        self.logits_v = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)
        self.logits_q = weight_norm(BCNet_q(v_num, x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """

        p = self.forward_all(v, q, v_mask)
        return p

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits_v = self.logits_v(v,q) # b x g x v x q
        logits_q=self.logits_q(v,q)
        #import pdb;pdb.set_trace()
        '''
        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(2).expand(logits_v.size())
            logits_v.data.masked_fill_(mask.data, -float('inf'))
        '''

        logits_v=logits_v.contiguous()
        logits_q=logits_q.contiguous()
        v_att = nn.functional.softmax(logits_v.view(-1, v_num * 2048), 1)
        q_att = nn.functional.softmax(logits_v.view(-1, q_num * 2048), 1)
        v_att=v_att*2048
        q_att=q_att*2048
        #import pdb;pdb.set_trace()
        return v_att.view(-1, v_num, 2048),q_att.view(-1,q_num,2048)

class BiAttention2(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention2, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(outerproduct(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p = self.forward_all(v, q, v_mask)
        return p

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q
        logits=logits/1000
        #import pdb;pdb.set_trace()
        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(2).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))
        logits=logits.contiguous()
        
        p = nn.functional.softmax(logits.view(-1, v_num * 2048), 1)
        p=p*2000
        #import pdb;pdb.set_trace()
        return p.view(-1, v_num, 2048)