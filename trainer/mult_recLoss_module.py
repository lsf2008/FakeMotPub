import pytorch_lightning
import torch

import utils
from aemodel.loss import AeLoss
from aemodel.abnEval import AeScore
from sklearn.metrics import roc_auc_score
import torch.optim.lr_scheduler as lrs
import itertools

class MultRecLossModule(pytorch_lightning.LightningModule):
    def __init__(self, inputModel, **kwargs):
        super(MultRecLossModule, self).__init__()
        self.save_hyperparameters()
        self.model = inputModel
        # loss function
        layers = [0]
        self.aeLoss = AeLoss(layers)

        self.aeScore = AeScore(layers, batch_size= self.hparams.batch_size)
        # test results
        self.res={'maxAuc':0}


    def forward(self, x):
        self.model(x)

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

    def training_step(self, batch, batch_idx):
        x = batch['video']
        x = x.reshape((-1, *x.shape[2:]))

        x_r, z, enc_out, dec_out = self(x)

        aeLss = self.aeLoss(x, x_r)
        logDic ={'aeLoss': aeLss}
        self.log(logDic, prog_bar=True)
        return aeLss

    def validation_step(self, batch, batch_idx):
        x = batch['video']
        y = batch['label']

        x_r, z, enc_out, dec_out = self(x)
        # calculte anomaly score
        aeScore = self.aeScore(x, x_r)

        return aeScore, y

    def validation_step_end(self, outputs):
        # obtain all scores and corresponding y
        scores, y = self.obtAScoresFrmOutputs(outputs)

        # compute auc, scores: dictory
        self.cmpCmbAUCWght(scores, y_true=y,
                           weight=self.hparams.cmbScoreWght, res=self.res)

        self.res['epoch'] = self.current_epoch
        self.log('auc', self.res['maxAuc'], prog_bar=True)



    # ====================== my functions=========================
    def obtAScoresFrmOutputs(self, outputs):
        '''
        # obtain all scores and corresponding y
        :param outputs: validation or test end outputs
        :return: dict{scores:}, label:list...

        '''
        aeScores = []
        y_true = []
        for i, out in enumerate(outputs):
            aeScore, y = out
            aeScores.extend(aeScore)
            y_true.extend(y)

        # normalize
        aeScores = utils.normalize(aeScores)
        scoreDic = {'aeScores': torch.tensor(aeScores)}

        return scoreDic, y_true

    def cmbScoreWght(self, scoreDic, weight=None):
        '''
        combine the scores with weight
        :param scoreDic: score dic {'aeScore': tensor}
        :param weight: list, lenght = len(scoreDic)
        :return: tensor of combined score
        '''
        itms = len(scoreDic)
        cmbScores = {}
        if weight==None:
            cmbScore = torch.zeros_like(list(scoreDic.values())[0])
            for i, k, v in enumerate(scoreDic.items()):
                cmbScore += v
            cmbScores[1] = cmbScore
        else:
            for p in itertools.combinations(weight, itms):
                cmbScore = torch.zeros_like(list(scoreDic.values())[0])
                for i, k, v in enumerate(scoreDic.items()):
                    cmbScore += v*p[i]
                cmbScores[p]=cmbScore
        return cmbScores


    def cmpCmbAUCWght(self, scoreDic, weight, y_true, res):
        '''
        compute AUC for one weight and score
        :param scoreDic: score dic {'aeScore': tensor}
        :param weight: list, lenght = len(scoreDic)
        :param y_true: groud truth
        :param res: dictory to store results
        :return: dictory
        '''
        cmbScores = self.cmbScoreWght(scoreDic, weight)

        for k, v in cmbScores.items():
            val_roc = roc_auc_score(y_true, v)

            if res['maxAuc'] > val_roc:
                res['maxAuc'] = val_roc
                res['coef'] = k
                res['label'] = y_true
                res['score'] = v
        return res
