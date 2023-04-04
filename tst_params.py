from aemodel.ae_multi_out import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer
from aemodel.autoencoder import convAE
import pytorch_lightning as pl
import time
import prettytable
from dataset.video_dataloader import VideoDataLoader
from trainer.mult_mot_recLoss_module import MultMotRecLossModule
from trainer.trainer import trainer_vd_module

pl.seed_everything(999999)
flg = 'ped2_mot'
if flg =='ped2_ae':
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
    with open('layers_tst_ae.txt', 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time-stat_time)/60} m')

if flg =='ped2_mot':
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

        vd = VideoDataLoader(**vars(args))
        # 模型
        model = AeMultiOut(input_shape=args.input_shape,
                           code_length=args.code_length)
        # module
        # ------------------only ae-----------------
        # mdl = MultAERecLossModule(model, **vars(args))

        # --------------------only motion--------------
        mdl = MultMotRecLossModule(model, **vars(args))

        # 使用module训练模型
        res = trainer_vd_module(args, mdl, vd)

        model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length)


        tbl.add_row([layer, res['maxAuc']])
    end_time = time.time()
    print(tbl)
    with open('layers_tst_mot.txt', 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time-stat_time)/60} m')