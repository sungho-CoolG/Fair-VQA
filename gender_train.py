"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import time
import itertools
import torch
import torch.nn as nn
import utils
import torch.optim.lr_scheduler as lr_scheduler


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss
##############################################################################

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots_labels=torch.zeros(*labels.size()).cuda()
    labels = torch.max(labels, 1)[1].data # argmax
    one_hots_labels.scatter_(1, labels.view(-1, 1), 1)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * one_hots_labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0):
    lr_default = 1e-3 if eval_loader is not None else 7e-4
    lr_decay_step = 2
    lr_decay_rate = 1
    lr_decay_epochs = range(10,20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 3
    grad_clip = .25

    utils.create_dir(output)
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
        if opt is None else opt
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))
    
   
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        print(N)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] =1e-3
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])
     
        '''
        for i, (v, b, q, a,ques,im,g,gender) in enumerate(train_loader):
            
            v = v.cuda()
            b = b.cuda()
            q = q.cuda()
            a = a.cuda()

            visual_pred, att = model(v, b, q, a)
            
            #import pdb;pdb.set_trace()
            gender=gender.squeeze(1)
            weights=torch.Tensor([2.0,1.0,0.0001]).cuda()
            #loss = instance_bce_with_logits(visual_pred, g.cuda())
            loss=nn.CrossEntropyLoss(weights)
            loss=loss(visual_pred,gender.cuda())
            #import pdb;pdb.set_trace()
            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()
           
            batch_score=torch.eq(visual_pred.argmax(1),gender.cuda()).sum()
            #batch_score = compute_score_with_logits(visual_pred, g.cuda()).sum()
            #total_loss += loss.item() * v.size(0)
            train_score += batch_score.item()
            #train_score+=batch_score
        '''
        total_loss /= N
        train_score = 100 * train_score / N
      
        if None != eval_loader:
            model.train(False)
            eval_score, bound, _ = evaluate(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        

     
        model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
        utils.save_model(model_path, model, epoch, optim)
           
        

@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = None
    import pickle as pkl
    from PIL import Image,ImageDraw
    lab2ans=pkl.load(open("./data/cache/trainval_label2ans.pkl",'rb'))
    woman=0
    woman_o=0
    man=0
    man_o=0
    other=0
    other_o=0
    #total_answer=torch.zeros(3129)
    for i, (v, b, q, a,ques,im,g,gender) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        gender=gender.squeeze(1)
        visual_pred, att = model(v, b, q, None)
        #batch_score=torch.eq(visual_pred.argmax(1),gender.cuda()).sum()
        
        #score+=batch_score
        batch_score = compute_score_with_logits(visual_pred, g.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += visual_pred.size(0)
        
        for j in range(len(v)):
            
            if gender[j]==0:
                woman=woman+1
                
                #if visual_pred[j].argmax()==0 or visual_pred[j].argmax()==1:
                if visual_pred[j].argmax()==gender[j].cuda():
                    woman_o=woman_o+1
                    '''
                    try:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/val2014/COCO_val2014_"+str(int(im[j])).zfill(12)+".jpg")
                    except:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/train2014/COCO_train2014_"+str(int(im[j])).zfill(12)+".jpg")
                    width,height=img.size
                    draw=ImageDraw.Draw(img)
                    topk_att,topk_idx=torch.topk(att,5,1)
                    
                    for k in range(3):
                        
                        
                        leftx=int(float(b[j][int(topk_idx[j][k])][0])*width)
                        lefty=int(float(b[j][int(topk_idx[j][k])][1])*height)
                        rightx=int(float(b[j][int(topk_idx[j][k])][2])*width)
                        righty=int(float(b[j][int(topk_idx[j][k])][3])*height)                    
                        coord=[leftx,lefty,leftx,righty,rightx,righty,rightx,lefty,leftx,lefty]
                        #draw.line(coord,'red',width=int(topk_att[j][k]*10))
                        draw.line(coord,'red',width=3)
                    if '/' not in str(ques[j]):
                        img.save("./gender_w_x/"+str(int(im[j]))+'_'+str(ques[j])+'_'+str(gender[j].item())+'_'+str(visual_pred[j].argmax().item())+".jpg")
                    ''' 
            elif gender[j]==1:
                #if visual_pred[j].argmax()==0 or visual_pred[j].argmax()==1:
                man=man+1
                
                if visual_pred[j].argmax()==gender[j].cuda():
                    man_o=man_o+1
                    '''
                    try:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/val2014/COCO_val2014_"+str(int(im[j])).zfill(12)+".jpg")
                    except:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/train2014/COCO_train2014_"+str(int(im[j])).zfill(12)+".jpg")
                    width,height=img.size
                    draw=ImageDraw.Draw(img)
                    topk_att,topk_idx=torch.topk(att,5,1)
                    
                    for k in range(3):
                        
                        
                        leftx=int(float(b[j][int(topk_idx[j][k])][0])*width)
                        lefty=int(float(b[j][int(topk_idx[j][k])][1])*height)
                        rightx=int(float(b[j][int(topk_idx[j][k])][2])*width)
                        righty=int(float(b[j][int(topk_idx[j][k])][3])*height)                    
                        coord=[leftx,lefty,leftx,righty,rightx,righty,rightx,lefty,leftx,lefty]
                        #draw.line(coord,'red',width=int(topk_att[j][k]*10))
                        draw.line(coord,'red',width=3)
                    if '/' not in str(ques[j]):
                        img.save("./gender_m_x/"+str(int(im[j]))+'_'+str(ques[j])+'_'+str(gender[j].item())+'_'+str(visual_pred[j].argmax().item())+".jpg")
                    '''
            else:
                #if visual_pred[j].argmax()==0 or visual_pred[j].argmax()==1:
                other=other+1
                
                if visual_pred[j].argmax()==gender[j].cuda():
                    other_o=other_o+1
                    '''
                    try:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/val2014/COCO_val2014_"+str(int(im[j])).zfill(12)+".jpg")
                    except:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/train2014/COCO_train2014_"+str(int(im[j])).zfill(12)+".jpg")
                    width,height=img.size
                    draw=ImageDraw.Draw(img)
                    topk_att,topk_idx=torch.topk(att,5,1)
                    
                    for k in range(3):
                        
                        
                        leftx=int(float(b[j][int(topk_idx[j][k])][0])*width)
                        lefty=int(float(b[j][int(topk_idx[j][k])][1])*height)
                        rightx=int(float(b[j][int(topk_idx[j][k])][2])*width)
                        righty=int(float(b[j][int(topk_idx[j][k])][3])*height)                    
                        coord=[leftx,lefty,leftx,righty,rightx,righty,rightx,lefty,leftx,lefty]
                        #draw.line(coord,'red',width=int(topk_att[j][k]*10))
                        draw.line(coord,'red',width=3)
                    if '/' not in str(ques[j]):
                        img.save("./gender_o_x/"+str(int(im[j]))+'_'+str(ques[j])+'_'+str(gender[j].item())+'_'+str(visual_pred[j].argmax().item())+".jpg")
                    '''
       
    score = float(score) / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    
    woman_score=float(woman_o)/woman
    man_score=float(man_o)/man
    other_score=float(other_o)/other
    print(woman_score*100)
    print(man_score*100)
    print(other_score*100)
    print(woman_score*1/3+man_score*1/3+other_score*1/3)
    return score, upper_bound, None

def calc_entropy(att): # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p+eps).log()).sum(2).sum(0) # g
