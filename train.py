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


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    #import pdb;pdb.set_trace()
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
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
    
    woman=0
    woman_true=0
    woman_man=0
    woman_other=0
    man=0
    man_true=0
    man_woman=0
    man_other=0
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] =1e-3
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])
        
     
        import pickle as pkl
        from PIL import Image,ImageDraw
        lab2ans=pkl.load(open("./data/cache/trainval_label2ans.pkl",'rb'))
        '''
        for i, (v, b, q, a,ques,im,g,gender) in enumerate(train_loader):
          
            v = v.cuda()
            b = b.cuda()
            q = q.cuda()
            a = a.cuda()

            visual_pred, att = model(v, b, q, a)
           
      
            loss = instance_bce_with_logits(visual_pred, a)
            loss.backward()
            
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(visual_pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score.item()
            
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

        

        if (eval_loader is not None and eval_score > best_eval_score) or (eval_loader is None and epoch >= saving_epoch):
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score
        

@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = None

    woman=0
    woman_true=0
    woman_man=0
    woman_other=0
    man=0
    man_true=0
    man_woman=0
    man_other=0
    import pickle as pkl
    from PIL import Image,ImageDraw
    lab2ans=pkl.load(open("./data/cache/trainval_label2ans.pkl",'rb'))
    
    fr=open("race.txt",'w')
    fw=open("woman.txt",'w')
    fwx=open("woman_wrong.txt",'w')
    fwo=open("woman_true.txt",'w')
    fm=open("man.txt",'w')
    fmx=open("man_wrong.txt",'w')
    fmo=open("man_true.txt",'w')
    #total_answer=torch.zeros(3129)
    for i, (v, b, q, a,ques,im,g,gender) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        visual_pred, att = model(v, b, q, None)
        batch_score = compute_score_with_logits(visual_pred, a.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += visual_pred.size(0)
        woman_words=[' woman ',' women ',' female ',' girl ',' lady ',' she ',' her ',' hers ',' ladies ',' woman?',' women?',' female?',' girl?',' lady?',' she?',' her?',' hers?',' ladies?','Woman ','Women ','Female ','Girl ','Lady ','She ','Her ','Hers ','Ladies ',' woman\'s',' women\'s',' female\'s',' girl\'s',' lady\'s',' she\'s',' ladies\'',' woman\'s?',' women\'s?',' female\'s?',' girl\'s?',' lady\'s?',' she\'s?',' ladies\'?','Woman\'s','Women\'s','Female\'s','Girl\'s','Lady\'s','She\'s','Her\'s','Hers\'s','Ladies\'',' girls ','Girls ',' girls?',' girls\'','girls\'?']
        man_words=[' man ',' men ',' male ',' boy ',' he ',' his ',' gentleman ',' gentlemen ',' man?',' men?',' male?',' boy?',' he?',' his?',' gentleman?',' gentlemen?','Man ','Men ','Male ','Boy ','He ','His ','Gentleman ','Gentlemen ',' man\'s',' men\'s',' male\'s',' boy\'s',' his ',' gentleman\'s',' gentlemen\'s',' man\'s?',' men\'s?',' male\'s?',' boy\'s?',' gentleman\'s?',' gentlemen\'s?','Man\'s','Men\'s','Male\'s','Boy\'s','Gentleman\'s','Gentlemen\'s',' boys ','Boys ',' boys\'',' boys?',' boys\'?']

        #total_answer=total_answer+a.sum(0)
        for j in range(len(v)):
            '''
            if 'race' in ques[j] or 'ethnicity' in ques[j] or 'african' in lab2ans[int(a[j].argmax())] or 'asian' in lab2ans[int(a[j].argmax())] or 'american' in lab2ans[int(a[j].argmax())] or 'european' in lab2ans[int(a[j].argmax())] or 'white person' in lab2ans[int(a[j].argmax())] or 'black person' in lab2ans[int(a[j].argmax())]:
                fr.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                woman=woman+1
                
                fw.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                woman=woman+1
                if visual_pred[j].argmax().item()==a[j].argmax().item():
                    fwo.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                    woman_true=woman_true+1
                else:
                    fwx.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                    woman_other=woman_other+1
                
           '''
      
            for k in man_words:
                if k in ques[j] :
                    #import pdb;pdb.set_trace()
                    man=man+1
                    if visual_pred[j].argmax()==a[j].argmax().cuda():
                        man_true=man_true+1
                    else:
                        man_other=man_other+1

                    try:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/val2014/COCO_val2014_"+str(int(im[j])).zfill(12)+".jpg")
                    except:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/train2014/COCO_train2014_"+str(int(im[j])).zfill(12)+".jpg")
                    width,height=img.size
                    draw=ImageDraw.Draw(img)
                    topk_att,topk_idx=torch.topk(att,5,1)
                    
                    for k in range(5):
                        
                        
                        leftx=int(float(b[j][int(topk_idx[j][k])][0])*width)
                        lefty=int(float(b[j][int(topk_idx[j][k])][1])*height)
                        rightx=int(float(b[j][int(topk_idx[j][k])][2])*width)
                        righty=int(float(b[j][int(topk_idx[j][k])][3])*height)                    
                        coord=[leftx,lefty,leftx,righty,rightx,righty,rightx,lefty,leftx,lefty]
                        draw.line(coord,'red',width=int(topk_att[j][k]*10))
                
                    img.save("./man_question/"+str(int(im[j]))+'_'+str(ques[j])+'_'+str(lab2ans[int(a[j].argmax())])+'_'+lab2ans[int(visual_pred[j].argmax())]+".jpg")
            
            for k in woman_words:
                if k in ques[j] :
                    woman=woman+1
                    if visual_pred[j].argmax()==a[j].argmax().cuda():
                        woman_true=woman_true+1
                    else:
                        woman_other=woman_other+1
                    try:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/val2014/COCO_val2014_"+str(int(im[j])).zfill(12)+".jpg")
                    except:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/train2014/COCO_train2014_"+str(int(im[j])).zfill(12)+".jpg")
                    width,height=img.size
                    draw=ImageDraw.Draw(img)
                    topk_att,topk_idx=torch.topk(att,5,1)
                    
                    for k in range(5):
                        
                        
                        leftx=int(float(b[j][int(topk_idx[j][k])][0])*width)
                        lefty=int(float(b[j][int(topk_idx[j][k])][1])*height)
                        rightx=int(float(b[j][int(topk_idx[j][k])][2])*width)
                        righty=int(float(b[j][int(topk_idx[j][k])][3])*height)                    
                        coord=[leftx,lefty,leftx,righty,rightx,righty,rightx,lefty,leftx,lefty]
                        draw.line(coord,'red',width=int(topk_att[j][k]*10))
                
                    img.save("./man_question/"+str(int(im[j]))+'_'+str(ques[j])+'_'+str(lab2ans[int(a[j].argmax())])+'_'+lab2ans[int(visual_pred[j].argmax())]+".jpg")
        
        
          
            if lab2ans[int(a[j].argmax())]=='woman' or lab2ans[int(a[j].argmax())]=='women' or lab2ans[int(a[j].argmax())]=='female' or lab2ans[int(a[j].argmax())]=='girl':
                fw.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                woman=woman+1
                if lab2ans[int(visual_pred[j].argmax())]=='woman' or lab2ans[int(visual_pred[j].argmax())]=='women' or lab2ans[int(visual_pred[j].argmax())]=='female' or lab2ans[int(visual_pred[j].argmax())]=='girl':
                    woman_true=woman_true+1
                    fwo.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                elif lab2ans[int(visual_pred[j].argmax())]=='man' or lab2ans[int(visual_pred[j].argmax())]=='men' or lab2ans[int(visual_pred[j].argmax())]=='male' or lab2ans[int(visual_pred[j].argmax())]=='boy':
                    woman_man=woman_man+1
                    fwx.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                    try:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/val2014/COCO_val2014_"+str(int(im[j])).zfill(12)+".jpg")
                    except:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/train2014/COCO_train2014_"+str(int(im[j])).zfill(12)+".jpg")
                    width,height=img.size
                    draw=ImageDraw.Draw(img)
                    topk_att,topk_idx=torch.topk(att,5,1)
                    
                    for k in range(5):
                        
                        
                        leftx=int(float(b[j][int(topk_idx[j][k])][0])*width)
                        lefty=int(float(b[j][int(topk_idx[j][k])][1])*height)
                        rightx=int(float(b[j][int(topk_idx[j][k])][2])*width)
                        righty=int(float(b[j][int(topk_idx[j][k])][3])*height)                    
                        coord=[leftx,lefty,leftx,righty,rightx,righty,rightx,lefty,leftx,lefty]
                        draw.line(coord,'red',width=int(topk_att[j][k]*10))
                
                    img.save("./woman/"+str(int(im[j]))+'_'+str(ques[j])+'_'+str(lab2ans[int(a[j].argmax())])+'_'+lab2ans[int(visual_pred[j].argmax())]+".jpg")    
                else:
                    
                    woman_other=woman_other+1
                    fwx.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                
                
                

            if lab2ans[int(a[j].argmax())]=='man' or lab2ans[int(a[j].argmax())]=='men' or lab2ans[int(a[j].argmax())]=='male' or lab2ans[int(a[j].argmax())]=='boy':
                fm.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                man=man+1
                if lab2ans[int(visual_pred[j].argmax())]=='woman' or lab2ans[int(visual_pred[j].argmax())]=='women' or lab2ans[int(visual_pred[j].argmax())]=='female' or lab2ans[int(visual_pred[j].argmax())]=='girl':
                    man_woman=man_woman+1
                    fmx.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                    try:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/val2014/COCO_val2014_"+str(int(im[j])).zfill(12)+".jpg")
                    except:
                        img=Image.open("/media/cvpr-pu/1084447f-bfd3-49a1-814a-4fac086a5992/sungho/ban-vqa-master/tools/data/train2014/COCO_train2014_"+str(int(im[j])).zfill(12)+".jpg")
                    width,height=img.size
                    draw=ImageDraw.Draw(img)
                    topk_att,topk_idx=torch.topk(att,5,1)
                    
                    for k in range(5):
                        
                        
                        leftx=int(float(b[j][int(topk_idx[j][k])][0])*width)
                        lefty=int(float(b[j][int(topk_idx[j][k])][1])*height)
                        rightx=int(float(b[j][int(topk_idx[j][k])][2])*width)
                        righty=int(float(b[j][int(topk_idx[j][k])][3])*height)                    
                        coord=[leftx,lefty,leftx,righty,rightx,righty,rightx,lefty,leftx,lefty]
                        draw.line(coord,'red',width=int(topk_att[j][k]*10))
                
                    img.save("./man/"+str(int(im[j]))+'_'+str(ques[j])+'_'+str(lab2ans[int(a[j].argmax())])+'_'+lab2ans[int(visual_pred[j].argmax())]+".jpg")
                elif lab2ans[int(visual_pred[j].argmax())]=='man' or lab2ans[int(visual_pred[j].argmax())]=='men' or lab2ans[int(visual_pred[j].argmax())]=='male' or lab2ans[int(visual_pred[j].argmax())]=='boy':
                    man_true=man_true+1
                    fmo.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
                else:
                    man_other=man_other+1
                    fmx.write(str(ques[j]) + ' '+str(lab2ans[int(a[j].argmax())])+' '+str(lab2ans[int(visual_pred[j].argmax())])+'\n\n')
        
                
        import pdb;pdb.set_trace()        
         
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    fr.close()
    fw.close()
    fwx.close()
    fwo.close()
    fm.close()
    fmx.close()
    fmo.close()
    
    print("woman",woman)
    print("woman_true",woman_true)
    print("woman_man",woman_man)
    print("woman_other",woman_other)

    print("man",man)
    print("man_true",man_true)
    print("man_woman",man_woman)
    print("man_other",man_other)
    import pdb;pdb.set_trace()
    return score, upper_bound, None

def calc_entropy(att): # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p+eps).log()).sum(2).sum(0) # g
