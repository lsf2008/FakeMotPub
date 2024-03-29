
from dataset.video_dataloader import VideoDataLoader
import pytorch_lightning as pl
import time


def trainer_vd_module(args, module, vd, is_test=False):
    # vd = VideoDataLoader(**vars(args))
    train_dl = vd.train_dataloader()
    val_dl = vd.val_dataloader()

    # inmd = model(**vars(args))
    # mdl = MultRecLossModule(model, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # trainer.test(dataloaders=val_dl, ckpt_path='best')
    if is_test:
        start_time =time.time()
        trainer.test(dataloaders=val_dl, ckpt_path='best')
        end_time = time.time()
        print(f'test running time:{(end_time - start_time)}s')
    return module.res

