from aemodel.ae_multi_out_wghts import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer
from aemodel.autoencoder import convAE
import pytorch_lightning as pl
import time
import prettytable
from dataset.video_dataloader import VideoDataLoader
from trainer.mult_mot_recLoss_module import MultMotRecLossModule
from trainer.mult_ae_mot_recLoss_module import MultAeMotRecLossModule
from trainer.trainer import trainer_vd_module

flg = 'recLayers'

if flg == 'blkSize':
    input_shapes=[[3, 8, 32, 32], [3, 8, 56, 56], [3, 8, 64, 64], [3, 8, 24, 24]]
    pl.seed_everything(999999)
    stat_time = time.time()

    tbl = prettytable.PrettyTable()
    tbl.field_names=['shape', 'auc', 'cmb_coef', 'layers', 'epoch']
    for shape in input_shapes:
        args = initial_params('config/ave_cfg.yml')
        args.input_shape=shape

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

        tbl.add_row([shape, res['maxAuc'], res['coef'], args.rec_layers, res['epoch']])
    end_time = time.time()
    print(tbl)
    sv_name = 'data/ave/blk_size_cmp.txt'
    with open(sv_name, 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time - stat_time) / 60} m')

if flg == 'recLayers':
    print('recLayers'.center(100,'-'))
    layers=[[0], [0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4],
            [1],[1,2],[1,2,3], [1,2,3,4],
            [2], [2,3], [2,3,4],
            [3], [3,4]]
    pl.seed_everything(999999)
    stat_time = time.time()

    tbl = prettytable.PrettyTable()
    tbl.field_names=['recLayer', 'auc', 'cmb_coef', 'layers', 'epoch']
    for layer in layers:
        args = initial_params('config/ave_cfg.yml')
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

        tbl.add_row([layer, res['maxAuc'], res['coef'], args.wght_layers, res['epoch']])
    end_time = time.time()
    print(tbl)
    sv_name = 'data/ave/recLayer_cmp_wghtLayer[034].txt'
    with open(sv_name, 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time - stat_time) / 60} m')

if flg == 'token_size':
    print('recLayers'.center(100,'-'))
    layers=[[0], [0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4],
            [1],[1,2],[1,2,3], [1,2,3,4],
            [2], [2,3], [2,3,4],
            [3], [3,4]]
    pl.seed_everything(999999)
    stat_time = time.time()

    tbl = prettytable.PrettyTable()
    tbl.field_names=['recLayer', 'auc', 'cmb_coef', 'layers', 'epoch']
    for layer in layers:
        args = initial_params('config/ave_cfg.yml')
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

        tbl.add_row([layer, res['maxAuc'], res['coef'], args.wght_layers, res['epoch']])
    end_time = time.time()
    print(tbl)
    sv_name = 'data/ave/recLayer_cmp_wghtLayer[034].txt'
    with open(sv_name, 'w') as f:
        f.write(tbl.get_string())

    print(f'running time:{(end_time - stat_time) / 60} m')