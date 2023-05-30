# from aemodel.ae_multi_out import AeMultiOut
from aemodel.ae_multi_out_wght import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer
from aemodel.ae_mlp import Ae1Mlp2, Ae2Mlps, AeMlp, Ae1Mlp3,Ae3Mlps
import numpy as np
import torch
import pytorch_lightning as pl
import time
import prettytable
from dataset.video_dataloader import VideoDataLoader
from trainer.fakeMot_cross_module import FakeMotCrossModule
from trainer.Ae_fakeMot_cross_module import AeFakeMotCrossModule
from trainer.mult_ae_mot_recLoss_module import MultAeMotRecLossModule
from trainer.trainer import trainer_vd_module

flg = 'mlps'

if flg =='ped2_motLoss':
    # pl.seed_everything(555555)
    stat_time = time.time()
    print('ped2-motion'.center(100, '-'))
    tbl = prettytable.PrettyTable()
    tbl.field_names=['layers', 'auc', 'coef']
    layers = [[20,1,0], [20, 0, 0], [20, 1, 10],
              [0, 1, 10], [20, 0, 10]]

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
    with open('data/ped2/Ae1Mlp2_motLoss_ablation.txt', 'a') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time-stat_time)/60} m')

if flg =='ped2_mlps':
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
    with open('data/ped2/Ae1Mlp3_3layerMLP.txt', 'a') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time - stat_time) / 60} m')

if flg =='hiddenLayers':
    pl.seed_everything(999999)
    stat_time = time.time()
    print('ped2-motion'.center(100, '-'))
    tbl = prettytable.PrettyTable()
    tbl.field_names = ['mlpHiddenLayers', 'auc', 'coef']
    layers = [8, 16, 32, 64, 128, 256]

    for layer in layers:
        args = initial_params('config/ped2_cfg.yml')
        args.mlp_hidden = layer

        vd = VideoDataLoader(**vars(args))
        # 模型
        model = Ae1Mlp2(input_shape=args.input_shape,
                        code_length=args.code_length,
                        mlp_hidden=args.mlp_hidden)
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
    with open('data/ped2/Ae1Mlp2_hiddenLayers.txt', 'a') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time - stat_time) / 60} m')
if flg == 'blkSize':
    input_shapes = [[3, 8, 40, 40]]
    # pl.seed_everything(999999)
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

if flg == 'mlps':
    # pl.seed_everything(999999)
    from aemodel.model_utils import Loader

    loader = Loader('aemodel.ae_mlp')

    tbl = prettytable.PrettyTable()
    tbl.field_names = ['model','hiddenLayers', 'auc', 'coef']

    modelName = ['Ae0Mlp', 'Ae1Mlp2', 'Ae2Mlps', 'Ae3Mlps']
    layers = [8, 16, 32, 64, 128, 256]

    auc = 0

    for m in modelName:
        if m!='Ae0Mlp':

            for layer in layers:
                args = initial_params('config/ped2_cfg.yml')
                vd = VideoDataLoader(**vars(args))
                args.mlp_hidden= layer
                model = loader.get_instance(m, input_shape=args.input_shape,
                                code_length=args.code_length,
                                mlp_hidden=args.mlp_hidden)

                mdl = FakeMotCrossModule(model, **vars(args))
                res = trainer_vd_module(args, mdl, vd)

                tbl.add_row([m, layer, res['maxAuc'], res['coef']])
                if auc < res['maxAuc']:
                    auc = res['maxAuc']
                    tmp = res['maxAuc']
                    tmp = np.around(tmp, decimals=4)
                    torch.save(model.state_dict(), 'data/ped2_' + m + '_' + str(layer) + '_' +
                               str(tmp) + '.pt')
        else:
            args = initial_params('config/ped2_cfg.yml')
            vd = VideoDataLoader(**vars(args))
            model = loader.get_instance(m, input_shape=args.input_shape,
                                        code_length=args.code_length,
                                        mlp_hidden=args.mlp_hidden)

            mdl = FakeMotCrossModule(model, **vars(args))
            res = trainer_vd_module(args, mdl, vd)

            tbl.add_row([m, 0, res['maxAuc'], res['coef']])
            if auc < res['maxAuc']:
                auc = res['maxAuc']
                tmp = res['maxAuc']
                tmp = np.around(tmp, decimals=4)
                torch.save(model.state_dict(), 'data/ped2_' + m + '_'+str(0) + '_'+
                           str(tmp) + '.pt')

    print(tbl)
    with open('data/ped2/Ae1Mlp2_hiddenLayers.txt', 'a') as f:
        f.write(tbl.get_string())