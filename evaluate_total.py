"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import gender_base
import gender_fusion_model
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vqa', help='vqa or flickr')
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--gamma_base', type=int, default=4)
    parser.add_argument('--input', type=str, default='/media/cvpr-pu/4TB_1/sungho1/ban-vp-vqatest/saved_models/gender_fusion_notsignoid_1204')

    parser.add_argument('--input_base', type=str, default='/media/cvpr-pu/4TB_1/sungho1/ban-vp-vqatest/saved_models/gender_base_notsignoid_1204')
    parser.add_argument('--epoch', type=int, default=18)
    parser.add_argument('--epoch_base', type=int, default=23)
    parser.add_argument('--batch_size', type=int, default=512)




    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    if args.task == 'vqa':
        from gender_total_train import evaluate
        dict_path = 'data/dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        eval_dset = VQAFeatureDataset('val', dictionary, adaptive=True)

    elif args.task == 'flickr':
        from train_flickr import evaluate
        dict_path = 'data/flickr30k/dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        eval_dset = Flickr30kFeatureDataset('test', dictionary)
        args.op = ''
        args.gamma = 1

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(gender_fusion_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma, args.task).cuda()
    model_data = torch.load(args.input+'/model'+('_epoch%d' % args.epoch if 0 < args.epoch else '')+'.pth')

    model = model.cuda()
    model.load_state_dict(model_data.get('model_state', model_data))
    
    model_base = getattr(gender_base, constructor)(eval_dset, args.num_hid, args.op, args.gamma_base, args.task).cuda()
    model_base_data = torch.load(args.input_base+'/model'+('_epoch%d' % args.epoch_base if 0 < args.epoch_base else '')+'.pth')

    #model_base = nn.DataParallel(model_base).cuda()
    model_base.load_state_dict(model_base_data.get('model_state', model_base_data))
    



    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    model.train(False)
    model_base.train(False)

    eval_score, bound, entropy = evaluate(model,model_base, eval_loader)
    if args.task == 'vqa':
        print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    elif args.task == 'flickr':
        print('\teval score: %.2f/%.2f/%.2f (%.2f)' % (
        100 * eval_score[0], 100 * eval_score[1], 100 * eval_score[2], 100 * bound))


