import pytorch_lightning
import torch

import utils
from aemodel.loss import AeLoss, TimeGrdLoss, CrossEntropyLoss
from aemodel.abnEval import AeScore, TimeGrdScore
from sklearn.metrics import roc_auc_score
import torch.optim.lr_scheduler as lrs

from trainer import module_utils
# from aemodel.ae_mlp import AeMlp
class FakeMotCrossModule(pytorch_lightning.LightningModule):
    def __init__(self, inputModel, **kwargs):
        super(FakeMotCrossModule, self).__init__()
        self.save_hyperparameters(ignore='inputModel')
        self.model = inputModel
        # if not next(self.model.parameters()).is_cuda:
        #     self.model=self.model.cuda()
        # loss function

        layers = self.hparams.rec_layers
        # self.aeLoss = AeLoss(layers)
        self.motLoss = TimeGrdLoss(layers)
        self.crsLoss = CrossEntropyLoss()
        self.aeScore = AeScore(layers, batch_size=self.hparams.batch_size)
        self.motScore = TimeGrdScore(self.hparams.batch_size)
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
        x_mot_soft, x_mot_ae = self.model(x_all)

        # --------------compute the loss ---------------
        # normal reconstruction loss
        x_norm_mot, x_anorm_mot = torch.split(x_mot_ae, x.shape[0], dim=0)
        mot_rec_ls = self.motLoss(x, x_norm_mot)
        anorm_mot_ls = -self.motLoss(x, x_anorm_mot)
        # anorm_mot_ls=0

        # anomaly reconstruction loss
        cross_ls = self.crsLoss(x_mot_soft)

        # joint loss
        join_ls = (self.hparams.motLsAlpha[0]*mot_rec_ls +
                   self.hparams.motLsAlpha[1]*anorm_mot_ls +
                   self.hparams.motLsAlpha[2]*cross_ls)

        # print(f'------------x_r:{x_r.requires_grad},x:{x.requires_grad}--------------')
        logDic ={'mot_rec_ls': mot_rec_ls,
                 'cross_ls':cross_ls,
                 'anorm_mot_ls': anorm_mot_ls}
        self.log_dict(logDic, prog_bar=True)

        return join_ls

    def validation_step(self, batch, batch_idx):
        aeScore, y = self.tst_val_step(batch)
        return (aeScore, y)

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
        x_mot_soft, x_mot_rec = self.model(x)
        # x_r = self(x)

        # calculate anomaly score
        motScore = self.motScore(x, x_mot_rec)

        # --------------------prediction score----------------------
        # use the prediction as score directly.
        x_mot_softScore = x_mot_soft

        return motScore, x_mot_softScore, y

    def tst_val_step_end(self, outputs, logStr = 'val_roc'):
        # obtain all scores and corresponding y
        # scores, y = module_utils.obtAScoresFrmOutputs(outputs)
        scores,y = module_utils.obtMotAllScoresFrmOutputs(outputs)
        # self.res['epoch'] = self.current_epoch

        # compute auc, scores: dictory
        module_utils.cmpCmbAUCWght(scores, y_true=y,
                                   weight=self.hparams.cmbScoreWght,
                                   res=self.res,
                                   epoch=self.current_epoch)

        self.log(logStr, self.res['maxAuc'], prog_bar=True)
