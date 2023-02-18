
from dataset.video_dataloader import VideoDataLoader
from mult_recLoss_module import MultRecLossModule
import pytorch_lightning as pl

def train_trainer(args, model):
    vd = VideoDataLoader(**vars(args))
    train_dl = vd.train_dataloader()
    val_dl = vd.val_dataloader()

    mdl = MultRecLossModule(**vars(args), inputModel=model)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(mdl, train_dataloaders= train_dl,
                val_dataloaders= val_dl)
    return mdl.res

