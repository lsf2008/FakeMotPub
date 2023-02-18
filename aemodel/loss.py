#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py.py    
@Contact :   lishifeng2007@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/8 16:31   shifengLi     1.0         None
'''

# import lib
from aemodel.base import BaseModule
import torch
from aemodel.OneClsBase import OneClassBase
from torch import nn
import utils as utils

class AeBaseLoss(BaseModule):
    def __init__(self, stdInd):
        super(AeBaseLoss, self).__init__()
        self.stdInd = stdInd
    def forward(self, x, x_r):
        '''
        :param x:  list, each item is N*c*t*h*w
        :param x_r: corresponding list
        :return: list, each item is N, length=len(self.stdInd)
        '''

        if isinstance(x, list) and isinstance(x_r, list):
            x_r = list(reversed(x_r))
            enc_out = []
            dec_out = []
            for i in self.stdInd:
                enc_out.append(x[i])
                dec_out.append(x_r[i])

            ls = [(utils.cmpAeDiff(i, j)) for i, j in zip(enc_out, dec_out)]
        else:
            ls = utils.cmpAeDiff(x, x_r)
        return ls

class AeLoss(AeBaseLoss):
    def __init__(self, stdInd):
        super(AeLoss, self).__init__(stdInd)
        # self.stdInd = stdInd
    def forward(self, x, x_r):
        '''
        compute the reconstruction loss
        Parameters
        ----------
        x       orignal image (patches_per_batch, c, t, h,w )
        x_r     reconstruction image
        Returns
        -------
        '''
        # if isinstance(self.stdInd, int):
        #     if isinstance(x, torch.Tensor) and isinstance(x_r, torch.Tensor):
        #     # L = torch.sum(torch.pow((x - x_r), 2), dim=list(range(x.dim()-1, 0, -1)))
        #         L = utils.cmpAeDiff(x, x_r)
        #         ls = torch.mean(L)
        #
        #     if isinstance(x, list) and isinstance(x_r, list):
        #         x_r = list(reversed(x_r))
        #         enc_out=x[:self.stdInd]
        #         dec_out = x_r[:self.stdInd]
        #
        #         # if self.stdInd!=0:
        #         #     enc_out.append(x[0])
        #         #     dec_out.append(x_r[0])
        #         lss = [torch.mean(utils.cmpAeDiff(i, j)) for i, j in zip(enc_out, dec_out)]
        #
        #         ls = torch.mean(torch.tensor(lss))
        # if isinstance(self.stdInd, list):
        #     x_r = list(reversed(x_r))
        #     enc_out = []
        #     dec_out = []
        #     for i in self.stdInd:
        #         enc_out.append(x[i])
        #         dec_out.append(x_r[i])
        #
        #     lss = [(utils.cmpAeDiff(i, j)) for i, j in zip(enc_out, dec_out)]
        #
        #     ls = torch.mean(torch.tensor(lss))
        ls = super(AeLoss, self).forward(x, x_r)
        ls = [torch.mean(r) for r in ls]
        ls = torch.mean(torch.tensor(ls))
        return ls
        # return torch.max(L)

class GradientLoss(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_r):
        # Do padding to match the  result of the original tensorflow implementation
        '''
        param:
        x: (patches_per_batch, c, t, h,w )
        x_r: reconstruction
        '''

        # ls = utils.cmpGrdDiff(x, x_r)
        ls = utils.cmpGrdDiff(x, x_r)
        return torch.mean(ls)

class TimeGrdLoss(BaseModule):
    def __int__(self):
        super(TimeGrdLoss, self).__int__()

    def forward(self, x, x_r):
        '''
        x, x_r   patches_per_batch*3*8*h*w
        '''
        ls = utils.cmpTimGrdDiff(x, x_r)

        return torch.mean(ls)


class OneClassLoss(OneClassBase):
    def __init__(self, k, nu):
        super(OneClassLoss, self).__init__()
        self.k = k
        self.R = torch.zeros((k,))
        self.nu = nu

    def updtR(self):
        # dis = self.dis_kmean

        k = torch.unique(self.labels_)
        r = self.R.data.clone()
        for i in k:
            # lbk = lbs==i
            # if torch.sum(lbk==True)==0:
            #     continue
            r[i] = torch.quantile(torch.sqrt(self.dis_kmean[self.labels_ == i]), 1 - self.nu)
        # r=r.detach()
        self.R.data = r.data.clone()

    def cmpSkddRLoss(self, disR, lbs):
        '''compute R-svdd loss
        Parameters
        ----------
        dis : 1d tesor: 1*centers
            distance between samples and corresponding centers
        lbs : [1d tensor, samples]
            [labels belong to one center for each sample]
        R : [scalar]
            [scalar for svdd parameter]
        nu : [scalar]
            [scalar for svdd parameter ]
        device : [cpu/gpu], optional
            [description], by default None

        Returns
        -------
        [scalar]
            [all loss ]
        '''
        # lbs = self.labels_
        loss = 0.0
        k = torch.unique(lbs)
        for i in k:
            flg = lbs==i
            # # 该聚类中元素为0
            if torch.sum(flg ==True)==0:
                continue
            scorek = disR[flg]
            # print(scorek)
            # wght = torch.sum(lbs==i)
            rk = self.R[i]
            lossk = rk ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scorek), scorek))
            loss = loss + lossk
        return loss

    def cmpOneClssLoss(self, dis, lbs):
        '''compute the enegry for kmeans
        Parameters
        ----------
        dis : [2D tensor samples]
            minimum distance matrix obtained from obtClusterLbs fun
        lbs : [tensor, samples]
            samples labels

        '''
        # dis = self.dis_kmean
        # lbs = self.labels_
        loss = 0.0
        lb = torch.unique(lbs)
        for i in lb:
            loss = loss + torch.sum(dis[lbs == i]) / (torch.sum(lbs == i) + 1e-8)

        return loss

    def forward(self, cnters, z, objFlg='soft'):
        cnters = cnters.to(z.device)
        # R = R.to(dt.device)
        # 1, compute the distances to the centers
        dis = self.cmpDis(z, cnters)

        # 2. computer cluster labels
        self.labels_, self.dis_kmean = self.obtClusterLbs(dis)

        # 3. compute the loss for each cluster
        if objFlg == 'soft':
            # compute the square distance between R and samples
            disR = self.cmpSkddRDis(self.dis_kmean, self.labels_, self.R)

            loss_sv = self.cmpSkddRLoss(disR, self.labels_)
            # loss_sv = self.cmpOneClssRLoss()
        else:
            loss_sv = self.cmpOneClssLoss(self.dis_kmean, self.labels_)
        return loss_sv


class AeOneClsLoss(BaseModule):
    def __init__(self, k, objFlg='soft', lam=1):
        super(AeOneClsLoss, self).__init__()

        self.objFlg = objFlg

        self.lam=lam
        self.aeLoss = AeLoss()
        self.R = torch.zeros((k,))

        self.oneLoss = OneClassLoss(self.R)
    def forward(self, x, x_r, z, cnters):

        lss_ae = self.aeLoss(x, x_r)
        lss_one, dis, labels = self.oneLoss(cnters, z, self.objFlg)
        lss = lss_ae+self.lam*lss_one

        return lss, dis, labels


if __name__=='__main__':
    # x = torch.randn((12,3,8,40,40))
    # x_r = torch.randn((12,3,8,40,40))

    # tgl = TimeGrdLoss()
    # ael = AeLoss1()
    # print(ael(x,x_r))
    from aemodel.ae_multi_out import AeMultiOut
    input_shape = (3, 8, 32, 32)
    code_length = 64
    x = torch.randn((4, 3, 8, 32, 32))
    end = AeMultiOut(input_shape, code_length)
    x_r, z, enc_out, dec_out = end(x)
    # dec_out = reversed(dec_out)

    Aels = AeMultiLoss()
    print(Aels(enc_out, dec_out))