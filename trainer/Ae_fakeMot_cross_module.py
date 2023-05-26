import pytorch_lightning
import torch

import utils
from aemodel.loss import AeLoss, TimeGrdLoss, CrossEntropyLoss
from aemodel.abnEval import AeScore, TimeGrdScore, ClassScore
from sklearn.metrics import roc_auc_score
import torch.optim.lr_scheduler as lrs

from trainer import module_utils
# from aemodel.ae_mlp import AeMlp
class AeFakeMotCrossModule(pytorch_lightning.LightningModule):
    def __init__(self, inputModel, **kwargs):
        super(AeFakeMotCrossModule, self).__init__()
        self.save_hyperparameters(ignore='inputModel')
        self.model = inputModel
        # if not next(self.model.parameters()).is_cuda:
        #     self.model=self.model.cuda()
        # loss function

        layers = self.hparams.rec_layers
        self.aeLoss = AeLoss(layers)
        self.motLoss = TimeGrdLoss(layers)
        self.crsLoss = CrossEntropyLoss()
        self.aeScore = AeScore(layers, batch_size=self.hparams.batch_size)
        self.motScore = TimeGrdScore(self.hparams.batch_size)
        self.classScore = ClassScore(self.hparams.batch_size)
        # test results
        self.res={'maxAuc':0, 'coef':0}

    def forward(self, x):

        x_mot_soft, x_mot_rec = self.model(x)
        # x_r = self.model(x)
        # z = z.reshape(z.shape[0], -1)
        return x_mot_soft, x_mot_rec

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
        # x = module_utils.filterCrops(x)

        # -----------normal data reconstruction----------
        x = x.reshape((-1, *x.shape[2:]))
        # x_r, z, enc_out, dec_out = self.model(x)

        # -----------anomaly data reconstruction---------
        # shuffle the data along the time axis
        # x_shuffle = x[:, :, torch.randperm(x.size()[2]), :, :]
        x_shuffle = x[:, :, utils.shuffle_index(x.size()[2]), :, :]
        x_all = torch.cat((x, x_shuffle), dim=0)

        # b, c, t, h, w = x.shape
        x_mot_soft, x_mot_ae, x_app = self.model(x_all)

        # --------------compute the loss ---------------
        # normal reconstruction loss, norm motion + anormal motion
        x_norm_mot, x_anorm_mot = torch.split(x_mot_ae, x.shape[0], dim=0)

        mot_rec_ls = self.motLoss(x, x_norm_mot)
        anorm_mot_ls = -self.motLoss(x, x_anorm_mot)

        #--------------appearance loss--------------------
        # aeLoss = self.aeLoss(x_all, x_mot_ae)
        x_norm_app, x_anorm_app = torch.split(x_app, x.shape[0], dim=0)
        aeLoss1 = self.aeLoss(x, x_norm_app)

        # -------------anomaly reconstruction loss----------------
        cross_ls = self.crsLoss(x_mot_soft)

        # joint loss
        join_ls = (self.hparams.motLsAlpha[0]*mot_rec_ls +
                   self.hparams.motLsAlpha[1] * cross_ls +
                   self.hparams.motLsAlpha[2]*anorm_mot_ls +
                   self.hparams.motLsAlpha[3]*aeLoss1)

        # print(f'------------x_r:{x_r.requires_grad},x:{x.requires_grad}--------------')
        logDic ={'mot_rec_ls': mot_rec_ls,
                 'cross_ls': cross_ls,
                 'anorm_mot_ls': anorm_mot_ls,
                 'aeLoss1': aeLoss1}
        self.log_dict(logDic, prog_bar=True)

        return join_ls

    def validation_step(self, batch, batch_idx):
        scoreDic= self.tst_val_step(batch)
        return scoreDic

    def validation_epoch_end(self, outputs):
        self.tst_val_step_end(outputs, logStr='val_roc')
    def test_step(self, batch, batch_idx):
        return self.tst_val_step(batch)
    def test_epoch_end(self, outputs):
        return self.tst_val_step_end(outputs, logStr='tst_roc')

    # ====================== my functions=========================
    def tst_val_step(self, batch):
        x = batch['video']
        y = batch['label']
        # x = module_utils.filterCrops(x) # n, c, t, h, w
        x = x.reshape((-1, *x.shape[2:]))
        x_mot_soft, x_mot_rec, x_app = self.model(x)
        # x_r = self(x)

        # ------------------calculate anomaly score-----------------
        # motion score
        motScore = self.motScore(x, x_mot_rec)

        # prediction score, use the prediction as score directly.
        class_softScore = self.classScore(x_mot_soft)

        # appearance score
        aeScore = self.aeScore(x, x_app)

        scoreDic = {'classScores': class_softScore,
                    'motScores': motScore,
                    'aeScores': aeScore,
                    'label': y}
        return scoreDic

    def tst_val_step_end(self, outputs, logStr = 'val_roc'):
        # obtain all scores and corresponding y
        # scores, y = module_utils.obtAScoresFrmOutputs(outputs)
        scores, y = module_utils.obtAllScoresFrmDicOutputs(outputs)

        # compute auc, scores: dictory
        module_utils.cmpCmbAUCWght(scores, y_true=y,
                                   weight=self.hparams.cmbScoreWght,
                                   res=self.res,
                                   epoch=self.current_epoch)

        self.log(logStr, self.res['maxAuc'], prog_bar=True)
