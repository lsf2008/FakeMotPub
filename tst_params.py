from aemodel.ae_multi_out import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer
from aemodel.autoencoder import convAE
import pytorch_lightning as pl
import time
import prettytable

pl.seed_everything(999999)
flg = 'ped2'
if flg =='ped2':
    stat_time = time.time()

    tbl = prettytable.PrettyTable()
    tbl.field_names=['layers', 'auc']
    layers = [[0],
              [0,1], [0,1,2], [0,1, 2, 3], [0,1,2,3,4], [0,1, 3], [0,1, 4],
              [0,2], [0,2,3], [0,2,3,4],
              [0, 3], [0,3,4],
              [0, 4],
              [1], [1,2], [1,3],[1,4],
              [2],[2,3],[2,4],
              [3],[3,4],
              [2,3,4]
              ]
    # layers = [[0], [0,1,2, 3,4 ]]
    for layer in layers:
        args = initial_params('config/ped2_cfg.yml')
        args.layers=layer

        model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length)

        res = train_trainer(args, model)
        tbl.add_row([layer, res['maxAuc']])
    end_time = time.time()
    print(tbl)
    with open('layers_tst1.txt', 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time-stat_time)/60} m')