# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 20:19:23 2018

@author: Allen
"""

import zipfile
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as ply
import os
import sys
import imageio
from PIL import Image
import glob
import matplotlib.pyplot as plt
import time
import math
import datetime as dt
import pytz
import pickle
import logging
from io import BytesIO
import copy
from itertools import  filterfalse
from notebook import notebookapp
import urllib
import json
import os
import ipykernel
import Augmentor
from Augmentor.Operations import *
from Augmentor import *
import PIL
import cv2 as cv

global log

def get_logger(logger_name, level=logging.DEBUG):
    global log
    # logger
    file_name = '{}{}'.format('../salt_net/logs/',
                                logger_name)
    timestamp = dt.datetime.now(pytz.timezone('Australia/Melbourne'))\
        .strftime('%Y_%m_%d_%Hh')
    log_file = '{}_{}.log'.format(file_name, timestamp)
    logger = logging.getLogger(logger_name)

    formatter = (
        logging
        .Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   datefmt='%d/%m/%Y %H:%M:%S')
    )

    # for printing debug details
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(level)

    # for printing error messages
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.DEBUG)

    logger.setLevel(level)
    logger.handlers = []
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    log = logging.getLogger(logger_name)

    return log

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
else:
    dtype = torch.FloatTensor


class IOU_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        #print(y_pred.requires_grad)
        #y_pred = torch.where(y_pred.ge(0.5), torch.tensor(1.0), torch.tensor(0.0))
        i = y_pred.mul(y)
        u = (y_pred + y) - i
        mean_iou = torch.mean(i.view(i.shape[0],-1).sum(1) / u.view(i.shape[0],-1).sum(1))
        iou_loss = 1 - mean_iou
        #from boxx import g
        #g()

        return iou_loss


def load_all_data():
    try:
        print('Try loading data from npy and pickle files...')
        np_train_all = np.load('./data/np_train_all.npy')
        np_train_all_mask = np.load('./data/np_train_all_mask.npy')
        np_test = np.concatenate([np.load('./data/np_test_0.npy'), np.load('./data/np_test_1.npy')])
        with open('./data/misc_data.pickle', 'rb') as f:
            misc_data = pickle.load(f)
        print('Data loaded.')
        return (np_train_all, np_train_all_mask, np_test, misc_data)

    except:
        print('npy files not found. Reload data from raw images...')
        np_train_all, np_train_all_ids = load_img_to_np('./data/train/images')
        np_train_all_mask, np_train_all_mask_ids = load_img_to_np('./data/train/masks')
        df_train_all_depth = pd.read_csv('./data/depths.csv').set_index('id')
        np_test, np_test_ids = load_img_to_np('./data/test/images')
        np.save('./data/np_train_all.npy', np_train_all)
        np.save('./data/np_train_all_mask.npy', np_train_all_mask)
        for k, v in enumerate(np.split(np_test,2)):
            np.save(f'./data/np_test_{k}.npy', v)
        misc_data = {'df_train_all_depth': df_train_all_depth,
                     'np_train_all_ids': np_train_all_ids,
                     'np_train_all_mask_ids': np_train_all_mask_ids,
                     'np_test_ids': np_test_ids}
        with open('./data/misc_data.pickle', 'wb') as f:
            pickle.dump(misc_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data loaded.')
        return (np_train_all, np_train_all_mask, np_test, misc_data)


def rle_encoder2d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    s = pd.Series(x.clip(0,1).flatten('F'))
    s.index = s.index+1
    df = s.to_frame('pred').assign(zero_cumcnt=s.eq(0).cumsum())
    df = df.loc[df.pred.gt(0)]
    df_rle = df.reset_index().groupby('zero_cumcnt').agg({'index': min, 'pred': sum}).astype(int).astype(str)
    rle = ' '.join((df_rle['index'] + ' '+df_rle['pred']).tolist())

    return rle


def rle_encoder3d(x):
    return np.r_[[rle_encoder2d(e) for e in x]]


def load_img_to_np(img_path, num_channel=1):
    images = []
    img_ids = []
    for filename in sorted(glob.glob(f'{img_path}/*.png')): #assuming png
        img_id = filename.split('\\')[-1].split('.')[0]
        img_ids.append(img_id)
        images.append(np.array(imageio.imread(filename), dtype=np.uint8).reshape(101,101,-1)[:,:,0:num_channel])
    return (np.r_[images], img_ids)


def load_single_img(path, show=False):
    img = np.array(imageio.imread(path), dtype=np.uint8)
    if show:
        plt.imshow(img, cmap='gray')
    return img


def calc_raw_iou(a, b):
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().detach().numpy()
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    u = np.sum(np.clip(a+b, 0, 1), (1,2)).astype(np.float)
    i = np.sum(np.where((a+b)==2, 1, 0), (1,2)).astype(np.float)
    with np.errstate(divide='ignore',invalid='ignore'):
        iou = np.where(i==u, 1, np.where(u==0, 0, i/u))

    return iou


def calc_mean_iou(a, b):
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    iou = calc_raw_iou(a, b)
    iou_mean = (iou[:,None]>thresholds).mean(1).mean()

    return iou_mean


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_current_time_as_fname():
        timestamp = (
                dt.datetime.now(pytz.timezone('Australia/Melbourne'))
                .strftime('%Y_%m_%d_%H_%M_%S')
                )

        return timestamp


def plot_img_mask_pred(images, labels=None, img_per_line=8):
    images = [i.cpu().detach().numpy().squeeze() if isinstance(i, torch.Tensor) else i.squeeze() for i in images]
    num_img = len(images)
    if labels is None:
        labels = range(num_img)

    rows = np.ceil(num_img/img_per_line).astype(int)
    cols = min(img_per_line, num_img)
    f, axarr = plt.subplots(rows,cols)
    if rows==1:
        axarr = axarr.reshape(1,-1)
    f.set_figheight(3*min(img_per_line, num_img)//cols*rows)
    f.set_figwidth(3*min(img_per_line, num_img))
    for i in range(num_img):
        r = i//img_per_line
        c = np.mod(i,img_per_line)
        axarr[r,c].imshow(images[i], cmap='gray', vmin=0, vmax=1)
        axarr[r,c].grid()
        axarr[r,c].set_title(labels[i])

    plt.show()


def adjust_predictions(zero_mask_cut_off, X, y_pred, y=None):
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()
    y_pred_adj = y_pred.clip(0,1)

    # Set predictions to all 0 for black images
    black_img_mask = (X.mean((1,2,3)) == 0)
    y_pred_adj[black_img_mask]=0

    # set all predictions to 0 if the number of positive predictions is less than ZERO_MASK_CUTOFF
    y_pred_adj = np.r_[[e if e.sum()>zero_mask_cut_off else np.zeros_like(e) for e in y_pred_adj]]

    if y is not None:
        log.info(f'IOU score before: {calc_mean_iou(y_pred, y)}, IOU Score after:{calc_mean_iou(y_pred_adj, y)}')

    return y_pred_adj

def show_img_grid():
    pass
    #plt.imshow(torchvision.utils.make_grid(torch.from_numpy(y_train_black).unsqueeze(1)).permute(1, 2, 0))


def join_files(filePrefix, filePath, newFileName=None, returnFileObject=False, removeChunks=False):
    noOfChunks = int(glob.glob(f'{filePath}/{filePrefix}*')[0].split('-')[-1])
    dataList = []
    j = 0
    for i in range(0, noOfChunks, 1):
        j += 1
        chunkName = f"{filePath}/{filePrefix}-chunk-{j}-Of-{noOfChunks}"
        f = open(chunkName, 'rb')
        dataList.append(f.read())
        f.close()
        if removeChunks:
            os.remove(chunkName)

    if returnFileObject:
        fileOut = BytesIO()
        for data in dataList:
            fileOut.write(data)
        fileOut.seek(0)
        return fileOut
    else:
        fileOut = open(newFileName, 'wb')
        for data in dataList:
            fileOut.write(data)
        f2.close()
        print(f'File parts merged to {newFileName} successfully.')


# define the function to split the file into smaller chunks
def split_file_save(inputFile, outputFilePrefix, outputFolder, chunkSize=10000000):
    # read the contents of the file
    if isinstance(inputFile, BytesIO):
        data = inputFile.read()
        inputFile.close()
    else:
        f = open(inputFile, 'rb')
        data = f.read()
        f.close()

# get the length of data, ie size of the input file in bytes
    bytes = len(data)

# calculate the number of chunks to be created
    if sys.version_info.major == 3:
        noOfChunks = int(bytes / chunkSize)
    elif sys.version_info.major == 2:
        noOfChunks = bytes / chunkSize
    if(bytes % chunkSize):
        noOfChunks += 1

    chunkNames = []
    j = 0
    for i in range(0, bytes + 1, chunkSize):
        j += 1
        fn1 = f"{outputFilePrefix}-chunk-{j}-Of-{noOfChunks}"
        chunkNames.append(fn1)
        f = open(f'{outputFolder}/{fn1}', 'wb')
        f.write(data[i:i + chunkSize])
        f.close()

    return chunkNames




def save_model_state_to_chunks(epoch, model_state, optim_state, scheduler_state, stats, out_file_prefix, outputFolder, chunk_size=40000000):
    if out_file_prefix is None:
        return 'Model state is not saved as the out_file_prefix is None'

    state = {'epoch': epoch + 1,
             'model': model_state,
             'optimizer': optim_state,
             'scheduler': scheduler_state,
             'stats': stats}
    output = BytesIO()
    torch.save(state, output)
    output.seek(0)

    return split_file_save(output, out_file_prefix, outputFolder, chunkSize=chunk_size)


def train_model(model, dataloaders, criterion, optimizer, scheduler, model_save_name, other_data={},
                num_epochs=25, print_every=2, save_model_every=None, save_log_every=None, log=get_logger('SaltNet')):
    #args = locals()
    #args = {k:v.shape if isinstance(v, (torch.Tensor, np.ndarray)) else v for k,v in args.items()}
    #args = {k:v.shape if isinstance(v, (torch.Tensor, np.ndarray)) else v for k,v in args.items()}
    log.info('Start Training...')
    #log.info('Passed parameters: {}'.format(args))

    start = time.time()

    if torch.cuda.is_available():
        model.cuda()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_model = None
    best_iou = 0.0
    all_losses = []
    iter_count = 0
    X_train = other_data['X_train']
    X_val = other_data['X_val']
    y_train = other_data['y_train']
    y_val = other_data['y_val']
    X_train_mean_img = other_data['X_train_mean_img']

    for epoch in range(1, num_epochs+1):
        log.info('Epoch {}/{}'.format(epoch, num_epochs))
        log.info('-' * 20)
        if save_log_every is not None:
            if (epoch % save_log_every == 0):
                push_log_to_git()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_loss = []
            pred_vs_true_epoch = []

            for X_batch, y_batch, d_batch, X_id in dataloaders[phase]:
                #print(X_batch.shape)
                #print(len(iter(dataloaders[phase])))
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X_batch)
                    pred_vs_true_epoch.append([y_pred, y_batch])
                    #from boxx import g
                    #g()
                    loss = criterion(y_pred, y_batch.float())
                    all_losses.append(loss.item())
                    epoch_loss.append(loss.item())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        iter_count += 1
                if (phase == 'train') & (iter_count % print_every == 0):
                    iou_batch = calc_mean_iou(y_pred.ge(0.5), y_batch.float())
                    iou_acc = calc_clf_accuracy(y_pred.ge(0.5), y_batch.float())

                    log.info('Batch Loss: {:.4f}, Epoch loss: {:.4f}, Batch IOU: {:.4f}, Batch Acc: {:.4f} at iter {}, epoch {}, Time: {}'.format(
                        np.mean(all_losses[-print_every:]), np.mean(epoch_loss), iou_batch, iou_acc, iter_count, epoch, timeSince(start))
                    )
                    X_orig = X_train[X_id[0]].squeeze()
                    X_tsfm = X_batch[0,0].squeeze().cpu().detach().numpy()
                    X_tsfm = transform.resize(X_tsfm, (128, 128), mode='constant', preserve_range=True)
                    X_tsfm = X_tsfm[13:114,13:114] + X_train_mean_img.squeeze()
                    #X_tsfm = X_batch[0][X_batch[0].sum((1,2)).argmax()].squeeze().cpu().detach().numpy()[:101,:101] + X_train_mean_img.squeeze()

                    y_orig = y_train[X_id[0]].squeeze()
                    y_tsfm = (y_batch[0].squeeze().cpu().detach().numpy())
                    y_tsfm_pred =  y_pred[0].squeeze().gt(0.5)
                    plot_img_mask_pred([X_orig, X_tsfm, y_orig, y_tsfm, y_tsfm_pred],
                                       ['X Original', 'X Transformed', 'y Original', 'y Transformed', 'y Predicted'])

            y_pred_epoch = torch.cat([e[0] for e in pred_vs_true_epoch])
            y_true_epoch = torch.cat([e[1] for e in pred_vs_true_epoch])
            #from boxx import g
            #g()
            mean_iou_epoch = calc_mean_iou(y_pred_epoch.ge(0.5), y_true_epoch.float())
            mean_acc_epoch = calc_clf_accuracy(y_pred_epoch.ge(0.5), y_true_epoch.float())
            log.info('{} Mean IOU: {:.4f}, Mean Acc: {:.4f}, Best Val IOU: {:.4f} at epoch {}'.format(phase, mean_iou_epoch, mean_acc_epoch, best_iou, epoch))
            if phase == 'val' and mean_iou_epoch > best_iou:
                best_iou = mean_iou_epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                stats = {'best_iou': best_iou,
                         'all_losses': all_losses,
                         'iter_count': iter_count}
                log.info(save_model_state_to_chunks(epoch, copy.deepcopy(model.state_dict()),
                                                    copy.deepcopy(optimizer.state_dict()),
                                                    copy.deepcopy(scheduler.state_dict()), stats, model_save_name, '.'))
                best_model = (epoch, copy.deepcopy(model.state_dict()),
                                                    copy.deepcopy(optimizer.state_dict()),
                                                    copy.deepcopy(scheduler.state_dict()), stats, model_save_name, '.')
                log.info('Best Val Mean IOU so far: {}'.format(best_iou))
                # Visualize 1 val sample and predictions
                X_orig = X_val[X_id[0]].squeeze()
                y_orig = y_val[X_id[0]].squeeze()
                y_pred2 =  y_pred[0].squeeze().gt(0.5)
                plot_img_mask_pred([X_orig, y_orig, y_pred2],
                                   ['Val X Original', 'Val y Original', 'Val y Predicted'])
        if save_model_every is not None:
            if (epoch % save_model_every == 0) | (epoch == num_epochs-1):
                if best_model is not None:
                    log.info(save_model_state_to_chunks(*best_model))
                    push_model_to_git(ckp_name=model_save_name)
                    best_model = None
                else:
                    log.info("Skip pushing model to git as there's no improvement")

    # load best model weights
    model.load_state_dict(best_model_wts)
    log.info('-' * 20)
    time_elapsed = time.time() - start
    log.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log.info('Best val IOU: {:4f}'.format(best_iou))

    return model


def push_log_to_git():
    log.info('Pushing logs to git.')
    os.chdir('../salt_net')
    get_ipython().system("pwd")
    get_ipython().system("git config user.email 'allen.qin.au@gmail.com'")
    get_ipython().system('git add ./logs/*')
    get_ipython().system('git commit -m "Pushing logs to git"')
    get_ipython().system('git pull')
    get_ipython().system('git push https://allen.qin.au%40gmail.com:github0mygod@github.com/allen-q/salt_net.git --all')
    os.chdir('../salt_oil')
    #get_ipython().system('git filter-branch --force --index-filter "git rm --cached --ignore-unmatch *ckp*" --prune-empty --tag-name-filte

def push_log_to_git():
    log.info('Pushing logs to git.')
    os.chdir('../salt_net')
    get_ipython().system("pwd")
    get_ipython().system("git config user.email 'allen.qin.au@gmail.com'")
    get_ipython().system('git pull --no-edit')
    get_ipython().system('git add ./logs/*')
    get_ipython().system('git commit -m "Pushing logs to git"')
    get_ipython().system('git push https://allen.qin.au%40gmail.com:github0mygod@github.com/allen-q/salt_net.git --all')
    os.chdir('../salt_oil')

def push_model_to_git(ckp_name='ckp'):
    log.info('Pushing model state to git.')
    os.chdir('../salt_net')
    get_ipython().system("pwd")
    get_ipython().system("git config user.email 'allen.qin.au@gmail.com'")
    get_ipython().system('git pull --no-edit')
    get_ipython().system('git add .')
    get_ipython().system('git commit -m "save model state."')
    get_ipython().system('git push https://allen.qin.au%40gmail.com:github0mygod@github.com/allen-q/salt_net.git --all')
    #get_ipython().system(f'git filter-branch --force --index-filter "git rm --cached --ignore-unmatch *{ckp_name.split("/")[-1]}*" --prune-empty --tag-name-filter cat -- --all')
    os.chdir('../salt_oil')


def calc_clf_accuracy(a, b):
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().detach().numpy()
    acc = (a==b).sum()/a.size

    return acc


def dice_loss(input, target):
    smooth = 0.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class Dice_Loss(nn.Module):
    def __init__(self, smooth=1, alpha=1):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, inputs, targets):
        def _dice_loss(a, b):
            iflat = a.contiguous().view(1, -1)
            tflat = b.contiguous().view(1, -1)
            intersection = (iflat * tflat).sum()

            dice_loss = 1 - ((2. * intersection + self.smooth) /
                             (iflat.sum() + tflat.sum() + self.smooth))

            return dice_loss
        dice_loss = torch.stack([_dice_loss(a, b) for a,b in zip(inputs, targets)]).mean() * self.alpha

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets, per_image=True, ignore=None):
        lovasz_loss = self.lovasz_hinge(inputs, targets, per_image=per_image, ignore=ignore)

        return lovasz_loss

    def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        if per_image:
            loss = self.mean(self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                              for log, lab in zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.contiguous().view(-1)
        labels = labels.contiguous().view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum().float()
        #from boxx import g
        #g()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = filterfalse(np.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, inputs, targets):
        pos_y = torch.masked_select(inputs, targets.ge(0.5))
        neg_y = torch.masked_select(inputs, targets.lt(0.5))

        if pos_y.numel() > 0:
            pos_loss = F.relu(1-pos_y).mean()
        else:
            pos_loss = 0.

        if neg_y.numel() > 0:
            pos_loss = F.relu(neg_y + 1).mean()
        else:
            neg_loss = 0.

        loss = pos_loss + neg_loss
        return loss



def get_notebook_name():
    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for srv in notebookapp.list_running_servers():
        try:
            if srv['token']=='' and not srv['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(srv['url']+'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
            sessions = json.load(req)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return ''.join(sess['notebook']['name'].split('.')[:-1])
        except:
            pass  # There may be stale entries in the runtime directory
    return None


def adjust_brightness(img, alpha=None, beta=None):
    if alpha is None:
        # get a random num from 0.75 to 1.25
        alpha = (random.random()/2)+0.75
    if beta is None:
        # get a random num from -30 to 30
        beta = round((random.random()-0.5)*60)
    #print(f'a:{alpha}, b:{beta}')
    img_new = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img_new.reshape(img.shape)

class SaltDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, np_img, np_mask, df_depth, mean_img, out_size=101,
                 out_ch=1, transform=None, random_brightness=0):
        """
        Args:
            data_dir (string): Path to the image files.
            train (bool): Load train or test data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.np_img = np_img
        self.np_mask = np_mask.clip(0,1)
        self.df_depth = df_depth
        self.mean_img = mean_img
        self.out_size = out_size
        self.out_ch = out_ch
        self.transform = transform
        self.random_brightness = random_brightness

    def __len__(self):
        return len(self.np_img)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        X = self.np_img[idx]
        #X = X - self.mean_img

        if self.np_mask is None:
            y = np.zeros((101,101,1))
        else:
            y = self.np_mask[idx]

        if self.transform:
            img_in = PIL.Image.fromarray(np.c_[np.tile(X, 2), y*255])
            #img_in = PIL.Image.fromarray(np.tile(y, 3)*255)
            transformed = np.array(self.transform(img_in))
            #X = np.clip(transformed[:,:,0:1]/255, 0., 1.) - self.mean_img
            X = transformed[:,:,0:1]
            y = np.clip(transformed[:,:,2:3]/255, 0., 1.)

        if self.random_brightness > random.random():
            # disable brightness adjustment
            #X = adjust_brightness(X)
            X = np.clip(X/255, 0., 1.) - self.mean_img
        else:
            X = np.clip(X/255, 0., 1.) - self.mean_img
        #from boxx import g
        #g()
        X = np.moveaxis(X, -1,0)

        pad_size = self.out_size - X.shape[2]
        pad_first = pad_size//2
        pad_last = pad_size - pad_first
        X = np.pad(X, [(0, 0),(pad_first, pad_last), (pad_first, pad_last)], mode='reflect')

        d = self.df_depth.iloc[idx,0]

        X = torch.from_numpy(X).float().type(dtype)
        X = X.repeat(self.out_ch,1,1)
        y = transform.resize(y, (101, 101), mode='constant', preserve_range=True)
        y = torch.from_numpy(y).ge(0.5).float().squeeze().type(dtype)

        return (X,y,d,idx)


class Pipeline_Salt(Augmentor.Pipeline):
    def __init__(self, source_directory=None, output_directory="output", save_format=None):
        super(Pipeline_Salt, self).__init__(source_directory, output_directory, save_format)

    def torch_transform(self):
        def _transform(image):
            for operation in self.operations:
                r = round(random.uniform(0, 1), 1)
                if r <= operation.probability:
                    if not isinstance(image, list):
                        image = [image]
                    #print(type(operation))
                    #print(np.array(image[0]).shape)
                    image = operation.perform_operation(image)[0]

            return image


        return _transform

    def crop_random_align(self, probability, min_factor, max_factor, mask_diff_pct, resample_filter="BICUBIC"):
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif not (min_factor>0) and (min_factor<=1):
            raise ValueError("min_factor must be between 0 and 1.")
        elif not (max_factor>0) and (min_factor<=1):
            raise ValueError("max_factor must be between 0 and 1.")
        elif resample_filter not in Pipeline._legal_filters:
            raise ValueError("The save_filter argument must be one of %s." % Pipeline._legal_filters)
        else:
            self.add_operation(CropRandomAlign(probability, min_factor, max_factor, mask_diff_pct, resample_filter))

    def resize_random(self, probability, min_factor, max_factor, resample_filter="BILINEAR"):
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        elif resample_filter not in Pipeline._legal_filters:
            raise ValueError("The save_filter argument must be one of %s." % Pipeline._legal_filters)
        else:
            self.add_operation(ResizeRandom(probability=probability, min_factor=min_factor,
                                            max_factor=max_factor, resample_filter=resample_filter))
    def rotate_random_align(self, probability):
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(RotateRandomAlign(probability=probability))


class RotateRandomAlign(Operation):
    """
    This class is used to crop images a random factor between min_factor and max_factor and resize it to its original size.
    """
    def __init__(self, probability):
        Operation.__init__(self, probability)


    def perform_operation(self, images):
        def do(image):
            img_np = np.array(image)
            mask_in = img_np[:,:,2]
            mask_in_pct = (mask_in>0).sum()/mask_in.size
            #print(f'mask_in_pct: {mask_in_pct}')
            # if mask area is too small, do not rotate otherwise mask will become all black.
            if (mask_in_pct > 0) and (mask_in_pct <= 0.02):
                #print('No Change')
                return image
            elif mask_in_pct==0:
                #for iamge with black mask, rotate it up to 15 degree.
                mask_in_pct = 1
            # rotate between 2 to 15 degree based on mask area. Rotate more if mask area is bigger.
            rotation_bound = np.clip(mask_in_pct*100, 5, 15).round().astype(int)
            rotation = random.randint(2, rotation_bound)
            left_or_right = random.randint(0, 1)
            if left_or_right == 0:
                rotation = -rotation
            #print(f'rotation: {rotation}')
            # Get size before we rotate
            x = image.size[0]
            y = image.size[1]

            # Rotate, while expanding the canvas size
            image = image.rotate(rotation, expand=True, resample=Image.BILINEAR)

            # Get size after rotation, which includes the empty space
            X = image.size[0]
            Y = image.size[1]

            # Get our two angles needed for the calculation of the largest area
            angle_a = abs(rotation)
            angle_b = 90 - angle_a

            # Python deals in radians so get our radians
            angle_a_rad = math.radians(angle_a)
            angle_b_rad = math.radians(angle_b)

            # Calculate the sins
            angle_a_sin = math.sin(angle_a_rad)
            angle_b_sin = math.sin(angle_b_rad)

            # Find the maximum area of the rectangle that could be cropped
            E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
                (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
            E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
            B = X - E
            A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

            # Crop this area from the rotated image
            # image = image.crop((E, A, X - E, Y - A))
            image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

            # Return the image, re-sized to the size of the image passed originally
            return image.resize((x, y), resample=Image.BILINEAR)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


class ResizeRandom(Operation):
    """
    This class is used to resize an image by a random factor between min_factor and max_factor.
    """
    def __init__(self, probability, min_factor, max_factor, resample_filter="BICUBIC"):
        Operation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.resample_filter = resample_filter

    def perform_operation(self, images):
        """
        Resize the passed image and returns the resized image. Uses the
        parameters passed to the constructor to resize the passed image.

        :param images: The image to resize.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            width, height = image.size
            resize_factor = random.randrange(round(self.min_factor*100), round(self.max_factor*100), 1)/100
            width = round(width*resize_factor)
            height = round(height*resize_factor)
            print(f'New Width: {width}, New Height: {height}')
            return image.resize((width, height), eval("Image.%s" % self.resample_filter))

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images

class CropRandomAlign(Operation):
    """
    This class is used to crop images a random factor between min_factor and max_factor and resize it to its original size.
    """
    def __init__(self, probability, min_factor, max_factor, mask_diff_pct, resample_filter="BICUBIC"):
        Operation.__init__(self, probability)
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.mask_diff_pct = mask_diff_pct
        self.resample_filter = resample_filter

    def perform_operation(self, images):
        """
        Crop the passed :attr:`images` by percentage area, returning the crop as an
        image.

        :param images: The image(s) to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        resize_factor = random.randrange(round(self.min_factor*100), round(self.max_factor*100), 1)/100

        # The images must be of identical size, which is checked by Pipeline.ground_truth().
        w, h = images[0].size

        w_new = int(floor(w * resize_factor))  # TODO: Floor might return 0, so we need to check this.
        h_new = int(floor(h * resize_factor))

        def do(image, w, h):
            img_np = np.array(image)
            mask_in = img_np[:,:,2]
            mask_in_pct = (mask_in>0).sum()/mask_in.size
            img_out_candidate = None
            lowest_diff = 1
            for i in range(20):
                left_shift = random.randint(0, int((w - w_new)))
                down_shift = random.randint(0, int((h - h_new)))
                img_out = image.crop((left_shift, down_shift, w_new + left_shift, h_new + down_shift))
                mask_out = np.array(img_out)[:,:,2]
                mask_out_pct = (mask_out>0).sum()/mask_out.size
                #print(f'mask_in_pct:{mask_in_pct}, mask_out_pct:{mask_out_pct}')
                if (mask_in_pct==0) or (abs((mask_out_pct/mask_in_pct)-1) <= self.mask_diff_pct):
                    img_out_candidate = img_out
                    break
                if (abs((mask_out_pct/mask_in_pct)-1)) <= lowest_diff:
                    lowest_diff = abs((mask_out_pct/mask_in_pct)-1)
                    img_out_candidate = img_out
            if img_out_candidate is None:
                img_out_candidate = image
                print('Failed to crop image to fit requirements. Use orignal image.')
            #print(f'Image Size after crop:{img_out_candidate.size}')
            mask_out = np.array(img_out_candidate)[:,:,2]
            #print(f'image mask pct:{(mask_out>0).sum()/mask_out.size}')
            img_out_final = img_out_candidate.resize((w, h), eval("Image.%s" % self.resample_filter))
            #print(f'Image Size after resize:{img_out_final.size}')

            return img_out_final

        augmented_images = []

        for image in images:
            augmented_images.append(do(image, w, h))

        return augmented_images

def log_iter_stats(y_pred, y_batch, X_batch, X_id, train_params, other_data, epoch_losses, epoch, iter_count, start):
    #from boxx import g
    #g(),
    epoch_losses = [round(e.item(),4) for e in torch.stack(epoch_losses).mean(0)]
    iou_batch = calc_mean_iou(y_pred.ge(train_params['mask_cutoff']), y_batch)
    iou_acc = calc_clf_accuracy(y_pred.ge(train_params['mask_cutoff']), y_batch)

    log.info('Losses: {}, Batch IOU: {:.4f}, Batch Acc: {:.4f} at iter {}, epoch {}, Time: {}'.format(
            epoch_losses, iou_batch, iou_acc, iter_count, epoch, timeSince(start))
    )

    X_train = other_data['X_train']
    y_train = other_data['y_train']
    X_train_mean_img = other_data['X_train_mean_img']
    #print(all_losses)
    X_orig = X_train[X_id[0]].squeeze()/255
    X_tsfm = X_batch[0,0].squeeze().cpu().detach().numpy()
    X_tsfm = X_tsfm[13:114,13:114] + X_train_mean_img.squeeze()
    y_orig = y_train[X_id[0]].squeeze()
    y_tsfm = (y_batch[0].squeeze().cpu().detach().numpy())
    y_tsfm_pred =  y_pred[0].squeeze().gt(train_params['mask_cutoff'])
    plot_img_mask_pred([X_orig, X_tsfm, y_orig, y_tsfm, y_tsfm_pred],
                       ['X Original', 'X Transformed', 'y Original', 'y Transformed', 'y Predicted'])


def log_epoch_stats(model, optimizer, scheduler, y_pred, y_batch, X_batch, X_id, other_data, pred_vs_true_epoch, train_params, phase, epoch, iter_count, best_iou, all_losses, epoch_losses, best_model):
    y_pred_epoch = torch.cat([e[0] for e in pred_vs_true_epoch])
    y_true_epoch = torch.cat([e[1] for e in pred_vs_true_epoch])

    mean_iou_epoch = calc_mean_iou(y_pred_epoch.ge(train_params['mask_cutoff']), y_true_epoch.float())
    mean_acc_epoch = calc_clf_accuracy(y_pred_epoch.ge(train_params['mask_cutoff']), y_true_epoch.float())
    mean_loss_epoch = [round(e.item(),4) for e in torch.stack(epoch_losses).mean(0)]

    if phase == 'val':
        X_val = other_data['X_val']
        y_val = other_data['y_val']
        X_orig = X_val[X_id[0]].squeeze()/255
        y_orig = y_val[X_id[0]].squeeze()
        y_pred2 =  y_pred[0].squeeze().gt(train_params['mask_cutoff'])
        plot_img_mask_pred([X_orig, y_orig, y_pred2],
                           ['Val X Original', 'Val y Original', 'Val y Predicted'])
        if mean_iou_epoch > best_iou:
            best_iou = mean_iou_epoch
            stats = {'best_iou': best_iou,
                   'all_losses': all_losses,
                   'iter_count': iter_count}
            best_model = (epoch, copy.deepcopy(model.state_dict()),
                                              copy.deepcopy(optimizer.state_dict()),
                                              copy.deepcopy(scheduler.state_dict()), stats, train_params['model_save_name'], '.')
            log.info(save_model_state_to_chunks(*best_model))
            log.info('Best Val Mean IOU so far: {}'.format(best_iou))
        log.info('Val   IOU: {:.4f}, Acc: {:.4f}, Best Val IOU: {:.4f} at epoch {}'.format(mean_iou_epoch, mean_acc_epoch, best_iou, epoch))
    else:
        log.info('Train IOU: {:.4f}, Acc: {:.4f}, Loss: {} at epoch {}'.format(mean_iou_epoch, mean_acc_epoch, mean_loss_epoch, epoch))
        X_train = other_data['X_train']
        y_train = other_data['y_train']
        X_train_mean_img = other_data['X_train_mean_img']
        X_orig = X_train[X_id[0]].squeeze()/255
        X_tsfm = X_batch[0,0].squeeze().cpu().detach().numpy()
        X_tsfm = X_tsfm[13:114,13:114] + X_train_mean_img.squeeze()
        y_orig = y_train[X_id[0]].squeeze()
        y_tsfm = (y_batch[0].squeeze().cpu().detach().numpy())
        y_tsfm_pred =  y_pred[0].squeeze().gt(train_params['mask_cutoff'])
        plot_img_mask_pred([X_orig, X_tsfm, y_orig, y_tsfm, y_tsfm_pred],
                           ['X Original', 'X Transformed', 'y Original', 'y Transformed', 'y Predicted'])

    return best_iou, best_model

def save_model_to_git(epoch, train_params, num_epochs, prev_best_iou, best_iou, best_model):
    if (epoch % train_params['save_model_every']== 0) | (epoch == num_epochs-1):
        if train_params['model_save_name'] is None:
            log.info("Skip pushing model to git as model_save_name is None.")
        elif (best_model is not None) and (best_iou > prev_best_iou):
            log.info(save_model_state_to_chunks(*best_model))
            push_model_to_git(ckp_name=train_params['model_save_name'])
            prev_best_iou = best_iou
        else:
            log.info("Skip pushing model to git as there's no improvement")

    return prev_best_iou

def calc_loss(y_pred, y_batch, loss_fns, loss_fn_weights):
     losses = []
     for loss_fn, loss_fn_weight in zip(loss_fns, loss_fn_weights):
         loss = loss_fn_weight * loss_fn(y_pred, y_batch)
         losses.append(loss)

     return torch.stack(losses + [torch.stack(losses).sum()])

def train_model(model, dataloaders, loss_fns, loss_fn_weights, optimizer, scheduler, train_params, other_data):
    global log
    log = train_params['log']
    log.info('Start Training...')
    log.info((dataloaders, loss_fns, loss_fn_weights, optimizer, scheduler, train_params))
    num_epochs = train_params['num_epochs']
    start = time.time()
    if torch.cuda.is_available():
        model.cuda()
    best_model = None
    best_iou = 0.0
    prev_best_iou = train_params['model_save_iou_threshold']
    all_losses = []
    iter_count = 0

    for epoch in range(1, num_epochs+1):
        log.info('Epoch {}/{}'.format(epoch, num_epochs))
        log.info('-' * 20)
        if (epoch % train_params['save_log_every'] == 0):
            push_log_to_git()
        epoch_losses = []
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            pred_vs_true_epoch = []
            for X_batch, y_batch, d_batch, X_id in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X_batch)
                    pred_vs_true_epoch.append([y_pred, y_batch])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        losses = calc_loss(y_pred, y_batch.float(), loss_fns, loss_fn_weights)
                        epoch_losses.append(losses)
                        all_losses.append(losses)
                        loss = losses[-1]
                        loss.backward()
                        optimizer.step()
                        iter_count += 1
            best_iou, best_model = (
                    log_epoch_stats(model, optimizer, scheduler, y_pred,
                                    y_batch, X_batch, X_id, other_data,
                                    pred_vs_true_epoch, train_params,
                                    phase, epoch, iter_count, best_iou,
                                    all_losses, epoch_losses, best_model)
                    )

        prev_best_iou = save_model_to_git(epoch, train_params, num_epochs, prev_best_iou, best_iou, best_model)
        #from boxx import g
        #g()
        epoch_avg_loss = np.mean([e[-1].item() for e in epoch_losses])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            log.info(f'scheduler best: {scheduler.best} num_bad_epochs:{scheduler.num_bad_epochs}')
            scheduler.step(epoch_avg_loss)
            log.info([p['lr'] for p in optimizer.param_groups])
        else:
            scheduler.step(epoch)
            log.info(f"LR: {[round(p['lr'],4) for p in optimizer.param_groups]}")


    # load best model weights
    model.load_state_dict(best_model[1])
    log.info('-' * 20)
    log.info(f'Training complete in {(time.time() - start) // 60} mins. Best Val IOU {round(best_iou, 4)}')

    return model

class PolyLR(object):
    def __init__(self, optimizer, init_lr, lr_decay_iter=1, max_iter=150, power=0.9):
        super(PolyLR, self).__init__()
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.lr_decay_iter = lr_decay_iter
        self.max_iter = max_iter
        self.power = power

    def state_dict(self):
        return {}
    def step(self,iter):
        """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power
        """
        if iter % self.lr_decay_iter or iter > self.max_iter:
            return self.optimizer
        if not isinstance(self.init_lr, list):
            self.init_lr = [self.init_lr]
        for i in range(len(self.init_lr)):
            self.optimizer.param_groups[i]['lr'] = self.init_lr[i]*(1 - iter/self.max_iter)**self.power