import torch
import argparse
from dataset.video_dataloader import VideoDataLoader

from trainer.mult_ae_td_one_trainer import MulitAeTdOneTrainer
import pytorch_lightning as pl
import utils
# import pytorch_lightning.callbacks as plc

# pl.seed_everything(55555, workers=True)
def main_train(args, train_cfg, dt_cfg,
               trainer=MulitAeTdOneTrainer,
               sv_auc_pth='data/ped2_val_auc.pt'):

    print('---------test dataloader---------')
    vd = VideoDataLoader(**dt_cfg)
    train_dl = vd.train_dataloader()

    val_dl = vd.val_dataloader()
    # tst_dt = vd.test_dataloader()

    # for i, x in enumerate(test_dl):
    #     print(x['video'].shape)
    # print('-------test only Ae model--------')
    trainModel = trainer(**train_cfg)

    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(trainModel, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # torch.save(trainModel.res['auc'], sv_auc_pth)
    trainer.test(dataloaders=val_dl)
    return trainModel.res
def main_tst(args, dt_cfg,
             model=MulitAeTdOneTrainer,
             checkpoint_path="data/model/ped2_aetdsvdd.ckpt",
             hparams_file='data/model/ped2_aetdsvdd_hparams.yaml',
             sv_auc_pth='data/ped2_val_auc.pt'):

    print('---------test dataloader---------')
    vd = VideoDataLoader(**dt_cfg)
    # train_dl = vd.train_dataloader()

    # val_dl = vd.val_dataloader()
    tst_dt = vd.test_dataloader()

    trainer = pl.Trainer.from_argparse_args(args)

    Trained_model = model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file,
        map_location = None,
    )
    trainer.test(Trained_model, dataloaders=tst_dt)
    return Trained_model.res
    # torch.save(trainModel.res['auc'], sv_auc_pth)

if __name__=="__main__":
    args, train_cfg, dt_cfg = utils.initial_params(train_cfg='config/ave_train_cfg.yaml',
                                                   dt_cfg='config/dtped2_cfg.yml')
    res=main_tst(args, dt_cfg,
                 MulitAeTdOneTrainer,
                 checkpoint_path="data/model/ped2_aetdsvdd.ckpt",
                 hparams_file='data/model/ped2_aetdsvdd_hparams.yaml')
    print(res['score'])
