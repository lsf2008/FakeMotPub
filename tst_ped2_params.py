# from aemodel.ae_multi_out import AeMultiOut
from aemodel.ae_multi_out_wght import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer
from aemodel.ae_mlp import Ae1Mlp2, Ae2Mlps, AeMlp, Ae1Mlp3,Ae3Mlps

import pytorch_lightning as pl
import time
import prettytable
from dataset.video_dataloader import VideoDataLoader
from trainer.fakeMot_cross_module import FakeMotCrossModule
from trainer.Ae_fakeMot_cross_module import AeFakeMotCrossModule
from trainer.mult_ae_mot_recLoss_module import MultAeMotRecLossModule
from trainer.trainer import trainer_vd_module

flg = 'blkSize'

if flg =='ped2_mot':
    # pl.seed_everything(555555)
    stat_time = time.time()
    print('ped2-motion'.center(100, '-'))
    tbl = prettytable.PrettyTable()
    tbl.field_names=['layers', 'auc', 'coef']
    layers = [[25, 1, 10, 0.4],[20, 2, 10, 0.4]]

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
    with open('data/ped2/Ae2Mlps_randomSd.txt', 'a') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time-stat_time)/60} m')

if flg =='ped2_mot_ae':
    pl.seed_everything(999999)
    stat_time = time.time()
    print('ped2-motion-ae'.center(100, '-'))
    tbl = prettytable.PrettyTable()
    tbl.field_names = ['layers', 'auc', 'coef']
    layers = [[20, 1, 0.1, 0.5], [20, 1, 0, 0], [20, 1, 0.1, 0],
              [20, 1, 0.1, 0.3],
              [10, 1, 0.2, 0.3], [10, 1, 0.2, 0.5]]

    for layer in layers:
        args = initial_params('config/ped2_cfg.yml')
        args.motLsAlpha = layer

        vd = VideoDataLoader(**vars(args))
        # 模型
        model = Ae1Mlp3(input_shape=args.input_shape,
                        code_length=args.code_length)
        # module
        # ------------------only ae-----------------
        # mdl = MultAERecLossModule(model, **vars(args))

        # -------------------- motion--------------
        mdl = AeFakeMotCrossModule(model, **vars(args))

        # 使用module训练模型
        res = trainer_vd_module(args, mdl, vd)

        tbl.add_row([layer, res['maxAuc'], res['coef']])
    end_time = time.time()
    print(tbl)
    with open('data/ped2/AefakeMot_Ae1Mlp3_2layerMLP1.txt', 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time - stat_time) / 60} m')

if flg == 'blkSize':
    input_shapes = [[3, 8, 48, 48]]
    pl.seed_everything(999999)
    stat_time = time.time()

    tbl = prettytable.PrettyTable()
    tbl.field_names = ['shape', 'auc', 'cmb_coef', 'layers', 'epoch']
    for shape in input_shapes:
        args = initial_params('config/ped2_cfg.yml')
        args.input_shape = shape

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

        model = AeMultiOut(input_shape=args.input_shape,
                           code_length=args.code_length)

        tbl.add_row([shape, res['maxAuc'], res['coef'], args.rec_layers, res['epoch']])
    end_time = time.time()
    print(tbl)
    sv_name = 'data/ped2/blk_size_cmp.txt'
    with open(sv_name, 'a') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time - stat_time) / 60} m')


