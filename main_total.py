"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import gender_fusion_model
import gender_base
import utils
from utils import trim_collate
from dataset import tfidf_from_questions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vqa', help='vqa or flickr')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=4, help='glimpse')
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true', help='use visual genome dataset to train?')
    parser.add_argument('--tfidf', action='store_false', help='tfidf word embedding?')
    parser.add_argument('--input', type=str, default='/media/cvpr-pu/4TB_1/sungho1/ban-vp-vqatest/saved_models/gender_base/model_epoch17.pth')
    parser.add_argument('--input2', type=str, default='/media/cvpr-pu/4TB_1/sungho1/ban-vp-vqatest/saved_models/gender_fusion/model_epoch11.pth')
    parser.add_argument('--output', type=str, default='saved_models/gender_fusion_1204')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.task == 'vqa':
        from gender_total_train  import train
        dict_path = 'data/dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        train_dset = VQAFeatureDataset('train', dictionary, adaptive=True)
        val_dset = VQAFeatureDataset('val', dictionary, adaptive=True)
        w_emb_path = 'data/glove6b_init_300d.npy'


    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'args.txt'))
    logger.write(args.__repr__())

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(gender_base, constructor)(train_dset, args.num_hid, args.op, args.gamma, args.task).cuda()

    model2 = getattr(gender_fusion_model, constructor)(train_dset, args.num_hid, args.op, args.gamma, args.task).cuda()


    tfidf = None
    weights = None
    #import pdb;pdb.set_trace()

    model.w_emb.init_embedding(w_emb_path, tfidf, weights)
    model2.w_emb.init_embedding(w_emb_path, tfidf, weights)
    model=model.cuda()
    model2=model2.cuda()
    #model = nn.DataParallel(model).cuda()

    optim = None
    epoch = 0

    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1
    
    if args.input2 is not None:
        print('loading %s' % args.input2)
        model_data = torch.load(args.input2)
        model2.load_state_dict(model_data.get('model_state', model_data))



    if args.task == 'vqa':
        

        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4,collate_fn=utils.trim_collate)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=4,collate_fn=utils.trim_collate)
        #import pdb;pdb.set_trace()
    
    train(model,model2, train_loader, eval_loader, args.epochs, args.output, optim, epoch)
