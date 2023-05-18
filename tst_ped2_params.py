# from aemodel.ae_multi_out import AeMultiOut
from aemodel.ae_multi_out_wght import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer
from aemodel.ae_mlp import Ae1Mlp2, Ae2Mlps, AeMlp

import pytorch_lightning as pl
import time
import prettytable
from dataset.video_dataloader import VideoDataLoader
from trainer.fakeMot_cross_module import FakeMotCrossModule
from trainer.mult_ae_mot_recLoss_module import MultAeMotRecLossModule
from trainer.trainer import trainer_vd_module

flg = 'ped2_mot'

if flg =='ped2_mot':
    pl.seed_everything(999999)
    stat_time = time.time()
    print('ped2-motion'.center(100, '-'))
    tbl = prettytable.PrettyTable()
    tbl.field_names=['layers', 'auc', 'coef']
    layers = [[1,1,1], [15, 1, 1], [20, 1, 1], [10, 1, 1], [20, 1, 0.1],
              [20, 1, 0.5], [20, 1, 0]]

    for layer in layers:
        args = initial_params('config/ped2_cfg.yml')
        args.motLsAlpha=layer

        vd = VideoDataLoader(**vars(args))
        # 模型
        model = Ae2Mlps(input_shape=args.input_shape,
                           code_length=args.code_length)
        # module
        # ------------------only ae-----------------
        # mdl = MultAERecLossModule(model, **vars(args))

        # -------------------- motion--------------
        mdl = FakeMotCrossModule(model, **vars(args))

        # 使用module训练模型
        res = trainer_vd_module(args, mdl, vd)

        tbl.add_row([layer, res['maxAuc'], res['coef']])
    end_time = time.time()
    print(tbl)
    with open('data/ped2/fakeMot_Ae2Mlps_motSoftScore.txt', 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time-stat_time)/60} m')

if flg =='ped2_mot_ae':
    pl.seed_everything(999999)
    print('ae+motion'.center(100, '='))
    stat_time = time.time()

    tbl = prettytable.PrettyTable()
    tbl.field_names=['layers', 'auc', 'coef']
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
        args.rec_layers=layer

        vd = VideoDataLoader(**vars(args))
        # 模型
        model = AeMultiOut(input_shape=args.input_shape,
                           code_length=args.code_length,
                           layers=args.wght_layers)
        # module
        # ------------------only ae-----------------
        # mdl = MultAERecLossModule(model, **vars(args))

        # --------------------ae+ motion--------------
        mdl = MultAeMotRecLossModule(model, **vars(args))

        # 使用module训练模型
        res = trainer_vd_module(args, mdl, vd)

        model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length)


        tbl.add_row([layer, res['maxAuc'], res['coef']])
    end_time = time.time()
    print(tbl)
    with open('data/ped2/app+time/ped2_layers_ae_mot-layer012.txt', 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time-stat_time)/60} m')

if flg =='ped2_mot_ae_lossCoef':
    pl.seed_everything(999999)
    print('ped2_mot_ae_lossCoef'.center(100, '='))
    stat_time = time.time()

    tbl = prettytable.PrettyTable()
    tbl.field_names=['mot_coef', 'auc', 'cmb_coef', 'layers', 'epoch']
    coef = [0.1, 0.5, 1, 5, 10, 15, 20, 25,  30, 35, 40, 45]

    for layer in coef:
        args = initial_params('config/ped2_cfg.yml')
        args.motLsAlpha=layer

        vd = VideoDataLoader(**vars(args))
        # 模型
        model = AeMultiOut(input_shape=args.input_shape,
                           code_length=args.code_length,
                           layers=args.wght_layers)
        # module
        # ------------------only ae-----------------
        # mdl = MultAERecLossModule(model, **vars(args))

        # --------------------ae+ motion--------------
        mdl = MultAeMotRecLossModule(model, **vars(args))

        # 使用module训练模型
        res = trainer_vd_module(args, mdl, vd)

        model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length)


        tbl.add_row([layer, res['maxAuc'], res['coef'], args.rec_layers, res['epoch']])
    end_time = time.time()
    print(tbl)
    sv_name = 'data/ped2_34appTimeWightNormvar_AeMo_0td_layer'+str(args.rec_layers)+'_bsz'+str(args.input_shape[3])+'.txt'
    with open(sv_name, 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time-stat_time)/60} m')

