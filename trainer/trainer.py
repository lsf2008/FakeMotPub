
from dataset.video_dataloader import VideoDataLoader
from trainer.mult_ae_recLoss_module import MultRecLossModule
import pytorch_lightning as pl

def train_trainer(args, model):
    vd = VideoDataLoader(**vars(args))
    train_dl = vd.train_dataloader()
    val_dl = vd.val_dataloader()

    # inmd = model(**vars(args))
    mdl = MultRecLossModule(model, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(mdl, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # trainer.test(dataloaders=val_dl, ckpt_path='best')
    return mdl.res

def trainer_vd_module(args, module, vd):
    # vd = VideoDataLoader(**vars(args))
    train_dl = vd.train_dataloader()
    val_dl = vd.val_dataloader()

    # inmd = model(**vars(args))
    # mdl = MultRecLossModule(model, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # trainer.test(dataloaders=val_dl, ckpt_path='best')
    return module.res

