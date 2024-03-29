# from aemodel.ae_multi_out import AeMultiOut
import numpy as np
import torch
from config.param_seting import initial_params
from trainer.trainer import trainer_vd_module
from dataset.video_dataloader import VideoDataLoader
from aemodel.ae_mlp import AeMlp, Ae2Mlps, Ae1Mlp2,Ae1Mlp3
from trainer.fakeMot_cross_module import FakeMotCrossModule

import pytorch_lightning as pl
import prettytable as pt
import time
import utils as utils
from aemodel.model_utils import Loader

# pl.seed_everything(999999)
loader_model = Loader('aemodel.ae_mlp')
modelName = 'Ae1Mlp2'

flg = 'ped2'
tbl = pt.PrettyTable()
tbl.field_names = ['auc', 'cmb_coef', 'layers', 'epoch']
if flg =='ped2':
    pl.seed_everything(999999)
    # ===================ae==================
    stat_time = time.time()
    args = initial_params('config/ped2_cfg.yml')
    # 数据集
    vd = VideoDataLoader(**vars(args))
    # 模型
    model = loader_model.get_instance(modelName, input_shape=args.input_shape,
                                code_length=args.code_length,
                                mlp_hidden=args.mlp_hidden)
    # module
    # ------------------only ae-----------------
    # mdl = MultAERecLossModule(model, **vars(args))

    # -----------------only motion-----------------
    mdl = FakeMotCrossModule(model, **vars(args))

    # 使用module训练模型
    res = trainer_vd_module(args, mdl, vd, True)
    tmp = res['maxAuc']
    tmp = np.around(tmp, decimals=4)

    torch.save(model.state_dict(), 'data/ped2_stateDic'+
               str(tmp)+'.pt')
    end_time = time.time()
    tbl.add_row([res['maxAuc'], res['coef'], args.rec_layers, res['epoch']])
    print(f'running time:{(end_time - stat_time) / 60} m')

if flg =='ave':
    # ===================ae==================
    stat_time = time.time()
    args = initial_params('config/ave_cfg.yml')
    # 数据集
    vd = VideoDataLoader(**vars(args))
    # 模型
    model = Ae1Mlp2(input_shape=args.input_shape,
                    code_length=args.code_length)
    # module
    # ------------------only ae-----------------
    # mdl = MultAERecLossModule(model, **vars(args))

    # -----------------only motion-----------------
    mdl = FakeMotCrossModule(model, **vars(args))

    # 使用module训练模型
    res = trainer_vd_module(args, mdl, vd)
    end_time = time.time()
    tbl.add_row([res['maxAuc'], res['coef'], args.rec_layers, res['epoch']])
    print(f'running time:{(end_time - stat_time) / 60} m')


