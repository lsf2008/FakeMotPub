from aemodel.ae_multi_out import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer
from aemodel.autoencoder import convAE
import pytorch_lightning as pl
import time
pl.seed_everything(999999)
flg = 'ped2'
if flg =='ped2':
    stat_time = time.time()
    args = initial_params('config/ped2_cfg.yml')

    model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length)


    res = train_trainer(args, model)
    end_time = time.time()
    print(f'running time:{(end_time-stat_time)/60} m')

if flg =='ave':
    args = initial_params('config/ave_cfg.yml')

    model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length)
    res = train_trainer(args, model)
