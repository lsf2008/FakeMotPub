#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   abnEval.py.py    
@Contact :   lishifeng2007@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/12 10:58   shifengLi     1.0         None
'''

# import lib
from aemodel.OneClsBase import OneClassBase
import torch
from aemodel.base import BaseModule
import utils as utils
from aemodel.loss import AeBaseLoss


class PsnrScore(BaseModule):
    def __int__(self):
        super(PsnrScore, self).__int__()

    def forward(self, x, x_r):
        res = utils.psnr_error(x_r, x)
        return res


class AeScore(AeBaseLoss):
    def __init__(self, stdInd, batch_size):
        super(AeScore, self).__init__(stdInd)
        self.batch_size = batch_size

    def forward(self, x, x_r):
        '''
        :param x:   list, orignal image in each item (patches_per_batch, c, t, h,w )
        :param x_r:
        :return:
        return one scalar for the continue frames score
        '''
        # (b, h,w), only consider the last layer is better
        if isinstance(x, torch.Tensor) and isinstance(x_r, torch.Tensor):
            ls = utils.cmpAeDiff(x, x_r)
            ls = ls.reshape((self.batch_size, -1))
            ls, _ = torch.max(ls, dim=1)
        # if isinstance(x, list) and isinstance(x_r, list):
        #     x_r = list(reversed(x_r))
        #     '''
        #     lss=[[a1,a2,..a_batchsize],[b1,b2,..b_batchsize],[1,2,..batchsize]]
        #     len(lss)=6
        #     '''
        #     enc_out = x[self.stdInd:]
        #     enc_out.append(x[0])
        #     dec_out = x_r[self.stdInd:]
        #     dec_out.append(x_r[0])
        #
        #     all_ls = [(utils.cmpAeDiff(i, j)) for i, j in zip(enc_out, dec_out)]
        #     # 6*(sz_per_batch*b), each patch corresponding to 6 reconstruction error
        #     all_ls=torch.stack(all_ls)
        #
        #     # select the maximum as the score among the 6 reconstruction errors
        #     all_ls, _ = torch.max(all_ls, dim=0)
        #
        #     all_ls = all_ls.reshape(self.batch_size, -1)
        #     ls, _ = torch.max(all_ls, dim=1)

        # list [N, N,..]
        # lss = [[a1, a2,..a_batchsize], [b1, b2,..b_batchsize], [1, 2,..batchsize]]
        # lss = super(AeScore, self).forward(x, x_r)
        # if isinstance(lss, list):
        #     all_ls = torch.stack(lss)
        #
        #     all_ls, _ = torch.max(all_ls, dim=0)
        #     all_ls = all_ls.reshape(self.batch_size, -1)
        #     ls, _ = torch.max(all_ls, dim=1)
        # else:
        #     ls = torch.tensor(0)
        return ls

class ClassScore(BaseModule):
    '''
    anormaly score using the output of prediction trained with crossEntropy
    '''
    def __init__(self, batch_size):
        super(ClassScore, self).__init__()
        self.batch_size = batch_size

    def forward(self, x):
        '''
        Parameters
        ----------
        x (b,2)
        Returns
        -------
        '''
        pred = x[:, 1]
        pred = pred.reshape((self.batch_size, -1))
        return torch.max(pred, dim=1)[0]

class TimeGrdScore(BaseModule):
    def __init__(self, batch_size):
        super(TimeGrdScore, self).__init__()
        self.batch_size = batch_size

    def forward(self, x, x_r):
        '''
        :param x: b*3*8*32*32
        :param x_r: b*3*8*32*32
        :return: vector (b,)
        '''
        ls = utils.cmpTimGrdDiff(x, x_r)
        ls = ls.reshape((self.batch_size, -1))
        ls = torch.max(ls, dim=-1)
        return ls[0]


class GdScore(BaseModule):
    def __init__(self, batch_size):
        super(GdScore, self).__init__()
        self.batch_size = batch_size

    def forward(self, x, x_r):
        '''

        :param x: b*3*8*32*32
        :param x_r: b*3*8*32*32
        :return: vector (b,)
        '''
        ls = utils.cmpGrdDiff(x, x_r)
        ls = ls.reshape((self.batch_size, -1))
        ls = torch.max(ls, dim=1)
        return ls[0]


class OneClsScore(OneClassBase):
    def __init__(self, batch_size):
        super(OneClsScore, self).__init__()
        self.batch_size = batch_size

    def forward(self, cnters, R, dt):
        cnters = cnters.to(dt.device)
        R = R.to(dt.device)
        # 1. compute distances between data and cnters
        dis = self.cmpDis(dt, cnters)

        # 2. compute the label of samples
        lbs, dismin = self.obtClusterLbs(dis)

        # 3. calculate subtraction between R and samples
        dis_k = self.cmpSkddRDis(dismin, lbs, R)

        dis_k = dis_k.reshape((self.batch_size, -1))
        dis_k = torch.max(dis_k, dim=-1)
        return dis_k[0]
    def cmpOneScore(self, cnters, dt):
        '''
        ||f(x)-c||^2
        :param cnters:  centers b,1
        :param dt:   (b*patches_per_batch)*code_length
        :return:
        '''
        cnters = cnters.to(dt.device)
        # 1. compute distances between data and cnters
        dis = self.cmpDis(dt, cnters)

        # 2. compute the label of samples
        lbs, dismin = self.obtClusterLbs(dis)

        dis_k = dismin.reshape((self.batch_size, -1))
        dis_k = torch.max(dis_k, dim=-1)
        return dis_k[0]


if __name__ == '__main__':
    x = torch.randn((12, 3, 8, 40, 40))
    x_r = torch.randn((12, 3, 8, 40, 40))


    # =============ae score==============
    # aes = AeScore(2)
    # gds = GdScore(2)
    # print(gds(x, x_r), aes(x, x_r))

    # ==============gradient score==========
    # grd = GdScore(2)
    # print(grd(x, x_r))

    # ==============timegradient score=============
    # tgrd = TimeGrdScore(3)
    # print(tgrd(x, x_r))
    # ==============oneclass score===========
    # R =torch.tensor([0.152,0.125])
    # cnter=torch.rand((2,64))
    # z=torch.rand((6,64))
    # oc = OneClsScore()
    # dis_k =oc(cnter, R, z)
    # ============class score===========
    x = torch.rand((7200, 2))
    cls1 = ClassScore(3)

    print(cls1(x))