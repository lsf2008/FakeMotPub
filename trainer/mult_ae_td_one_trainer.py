#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

@Contact :   lishifeng2007@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/9 21:40   shifengLi     1.0         None
'''

# import lib
import pytorch_lightning
import torch
import torch.optim.lr_scheduler as lrs

from aemodel.ae_multi_out import AeMultiOut, AeMultiOut1

from aemodel.loss import AeLoss, OneClassLoss, GradientLoss, TimeGrdLoss
from aemodel.abnEval import OneClsScore, GdScore, TimeGrdScore, AeScore
from sklearn.cluster import KMeans
from aemodel.baseline import cmb_ae_td_one_scores, getMaxAucFrmDict
import utils
from sklearn.metrics import roc_auc_score
import scipy.signal as signal
# import numpy as np
# torch.backends.cudnn.benchmark = False
class MulitAeTdOneTrainer(pytorch_lightning.LightningModule):
    def __init__(self, nu,
                 cnters=None,
                 objFlg='soft',
                 lr=1e-3,
                 weight_decay=1e-6,
                 updtR_epoches=2,
                 **kwargs):
        super(MulitAeTdOneTrainer, self).__init__()
        self.save_hyperparameters()

        # self.model = UCSDAE(input_shape=self.hparams.input_shape, code_length=self.hparams.code_length)

        # self.model = AvenueShAE(input_shape=self.hparams.input_shape, code_length=self.hparams.code_length)
        self.model = AeMultiOut(input_shape=self.hparams.input_shape, code_length=self.hparams.code_length)

        # print('-initing-', self.model.named_parameters())
        self.cnters = torch.zeros((self.hparams.cntNum, self.hparams.code_length))
        # len(layer)=6
        layers = [0]
        self.aeLoss = AeLoss(layers)
        # self.gdLoss = GradientLoss()
        self.tgdLoss = TimeGrdLoss()
        self.oneClsLoss = OneClassLoss(self.hparams.cntNum, self.hparams.nu)

        self.aeScore = AeScore(layers, batch_size=self.hparams.batch_size)
        # self.grdScore = GdScore(batch_size=self.hparams.batch_size)
        self.timeGrdScore = TimeGrdScore(batch_size=self.hparams.batch_size)
        self.OneClsScore = OneClsScore(batch_size=self.hparams.batch_size)

        self.table = utils.initTable(['epoch', 'aeCoef', 'tdgCoef', 'oneCoef', 'AUC'])

        self.res={'maxAuc':0,'epoch':0, 'coef':0, 'auc':[],'score': [], 'label':[]}
        # self.res=auc: auc for each v
            # self.cnters_dt=[]

    def forward(self, x):
        # x_r=(1040, 3, 8, 32, 32)
        x_r, z, enc_out, dec_out = self.model(x)  # z(1040, 1, 64)
        z = z.reshape(z.shape[0], -1)
        return x_r, z, enc_out, dec_out

    def aeWarmTrain(self, x):
        '''
        :param x:   b, batches, c, t, h, w
        :return:
        '''
        # x = x.reshape((-1, *x.shape[2:]))  # (1040, 3, 8, 32, 32)
        x_r, z, enc_out, dec_out = self(x)
        # gLoss = self.gdLoss(x, x_r)
        aloss = self.aeLoss(x, x_r)
        tgdLoss = self.tgdLoss(x, x_r)

        return aloss*self.hparams.aeLsAlpha +\
               tgdLoss*self.hparams.tgLsAlpha

    def training_step(self, batch, batch_idx):
        x = batch['video']
        # batch_size = x.shape[0]
        x = x.reshape((-1, *x.shape[2:]))  # (1040, 3, 8, 32, 32)

        if self.trainer.current_epoch<self.hparams.aeWarmEpochs:
            aeLs = self.aeWarmTrain(x)
            self.log('3-aeLoss', aeLs, prog_bar= True)
            return aeLs

        x_r, z, enc_out, dec_out = self(x)
        if torch.sum(self.cnters) == 0:
            # training centers
            dt = z.detach().data.clone().cpu()
            km = KMeans(n_clusters=self.hparams.cntNum, init='k-means++')
            km.fit(dt)
            self.cnters = torch.tensor(km.cluster_centers_, dtype=z.dtype).to(self.device)

        # loss
        aeLoss = self.aeLoss(enc_out, dec_out)
        # gdLoss = self.gdLoss(x, x_r)
        tgdLoss = self.tgdLoss(x, x_r)
        oneLoss = self.oneClsLoss(self.cnters, z, self.hparams.objFlg)

        allLoss = aeLoss*self.hparams.aeLsAlpha + \
                  tgdLoss*self.hparams.tgLsAlpha + \
                  self.hparams.oneLsAlpha * oneLoss

        if self.hparams.objFlg=='soft' and self.trainer.current_epoch >= self.hparams.updtR_epoches:
            self.oneClsLoss.updtR()

        if self.hparams.objFlg == 'soft':
            logDic = {self.__class__.__name__+':oneLs': oneLoss,'aeLs':aeLoss,
                  'tLs': tgdLoss,
                  'R':self.oneClsLoss.R[0]}
        else:
            logDic = {self.__class__.__name__ + 'oneLoss': oneLoss, 'aeLoss': aeLoss,
                      'tLoss': tgdLoss}
        self.log_dict(logDic, prog_bar=True)

        return allLoss

    def calAbnSepPSNR(self, x):
        '''
        :param x: (b, patch_num_per_batch, 3, 8, 32, 32)
        :return:
        '''
        b, p, c, t, h, w = x.shape
        xp = x.reshape((-1, *x.shape[2:]))
        x_r, z = self(xp)
        x_r = x_r.reshape(*x.shape)

        # b*1
        p = utils.psnr_error(x_r,x)
        return p
    def calAbnSepScore(self, x):
        '''
        params:
        x: (b, patch_num_per_batch, 3, 8, 32, 32)
        return:
        1d tensor of anormaly scores
        '''

        # bt_sz = x.shape[0]
        x = x.reshape((-1, *x.shape[2:]))  # (1040, 3, 8, 32, 32)
        x_r, z, enc_out, dec_out = self(x)
        aeScore = self.aeScore(x, x_r)
        # gdScore = self.grdScore(x, x_r)
        tdgScore = self.timeGrdScore(x, x_r)
        # oneScore = self.OneClsScore(self.cnters, self.oneClsLoss.R, z)
        oneScore = self.OneClsScore.cmpOneScore(self.cnters, z)
        # oneScore = utils.normalize(oneScore)
        # score = aeScore+oneScore

        return aeScore, tdgScore, oneScore
    def tst_val_step(self, batch):
        x = batch['video']
        y = batch['label']

        s = self.calAbnSepPSNR(x)
        return s, y

    def tst_val_end(self, outputs, logFlg='val_auc'):
        s=[]
        y_gt=[]
        for x,y in outputs:
            s.extend(x.detach().cpu().numpy())
            y_gt.extend(y.detach().cpu().numpy())

        # s = utils.normalize(torch.tensor(s))
        if len(y_gt) > 1 and len(s) > 1:
            y_gt[-1] = 1
            val_roc = roc_auc_score(y_gt, s)
        else:
            val_roc = -1
        self.log(logFlg, val_roc, prog_bar=True)

        return s, y_gt

    def tst_val_step_fun(self, batch):
        x = batch['video']
        y = batch['label']
        # x = x.reshape((-1, *x.shape[2:]))  # (1040, 3, 8, 32, 32)

        aeScore, tdgScore, oneScore = self.calAbnSepScore(x)

        return aeScore, tdgScore, oneScore, y
    def tst_val_end_fun(self, outputs, logFlg= 'val_roc'):
        # preds is dict
        preds, y = self.calAbnScore(outputs)

        self.res, auc = getMaxAucFrmDict(y, preds, self.res, self.current_epoch)
        if self.current_epoch==0:
            self.res['maxAuc']=0

        curEpoch = str(self.res['epoch'])+' epoch '
        print('\n'+curEpoch+f' coef:{self.res["coef"]}, maxauc:{self.res["maxAuc"]}')

        self.log(logFlg, auc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # print('validation step')
        aeScore,  tgdScore, oneScore, y= self.tst_val_step_fun(batch)
        return (aeScore, tgdScore, oneScore, y)
        # s, y = self.tst_val_step(batch) # 效果不好
        # return s,y

    def validation_epoch_end(self, outputs):
        # print('validation step end')
        self.tst_val_end_fun(outputs, logFlg='val_roc')
        # self.tst_val_end(outputs, logFlg='val_roc')

        # if self.current_epoch==self.hparams.max_epochs-1 and self.res:
        #     sv_params = 'result/'+list(self.res.keys())[0]+'-'+str(self.hparams.max_epochs)+'.txt'
        #     with open(sv_params, 'w+') as f:
        #         f.write(str(self.table))
        #         f.close()
    def calAbnScore(self, outputs):
        aeS = []
        tgdS = []
        oneS = []
        y_true = []
        for i, p in enumerate(outputs):
            aeScore, tgdScore, oneScore, y = p
            aeS.extend(aeScore.detach().cpu().numpy())
            # gdS.extend(gdScore.detach().cpu().numpy())
            tgdS.extend(tgdScore.detach().cpu().numpy())
            oneS.extend(oneScore.detach().cpu().numpy())
            y_true.extend(y.detach().cpu().numpy())
            # print(f'{i}-ae:{len(aeScore)},gd:{len(gdScore)}, tgd:{len(tgdScore)}, oneS:{len(oneScore)}')

        aeS = utils.normalize(torch.tensor(aeS))
        # gdS = utils.normalize(torch.tensor(gdS))
        tgdS = utils.normalize(torch.tensor(tgdS))
        oneS = utils.normalize(torch.tensor(oneS))
        if aeS!=None and oneS!=None and tgdS!=None:
            # aeS = signal.medfilt(aeS, 5)
            # tgdS = signal.medfilt(tgdS, 5)
            # oneS = signal.medfilt(oneS, 5)
            '''
            s is a dict
            '''
            s = cmb_ae_td_one_scores(tgdS, aeS, oneS,
                                self.hparams.tgScoreAlpha,
                                self.hparams.aeScoreAlpha,
                                self.hparams.oneScoreAlpha)

            # s = utils.normalize(s) #no normalization is better
            return s, y_true
        else:
            return {'0':0}, [0]


    def test_step(self, batch, batch_idx):
        return self.tst_val_step_fun(batch)
        # return self.tst_val_step(batch)

    def test_epoch_end(self, outputs):
        self.tst_val_end_fun(outputs, logFlg='tstAuc')
        # self.tst_val_end(outputs, logFlg='tst_auc')
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalide lr_scheduler type!')
            return [optimizer], [scheduler]
