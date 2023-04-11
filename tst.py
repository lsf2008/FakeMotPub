from aemodel.ae_multi_out import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer,trainer_vd_module
# from aemodel.autoencoder import convAE
from dataset.video_dataloader import VideoDataLoader
from trainer.mult_ae_recLoss_module import MultAERecLossModule
from trainer.mult_mot_recLoss_module import MultMotRecLossModule
from trainer.mult_ae_mot_recLoss_module import MultAeMotRecLossModule
from trainer.mult_recLoss_module_finch import MultRecLossModuleFinch
import pytorch_lightning as pl
import time
pl.seed_everything(999999)
flg = 'ped2'

if flg =='ped2':
    # ===================ae==================
    stat_time = time.time()
    args = initial_params('config/ped2_cfg.yml')
    # 数据集
    vd = VideoDataLoader(**vars(args))
    # 模型
    model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length)
    # module
    # ------------------only ae-----------------
    # mdl = MultAERecLossModule(model, **vars(args))

    # --------------------only motion--------------
    mdl = MultAeMotRecLossModule(model, **vars(args))

    # 使用module训练模型
    res = trainer_vd_module(args, mdl, vd)
    end_time = time.time()
    print(f'running time:{(end_time - stat_time) / 60} m')

    # ===================AE+ FINCH==================
    # stat_time = time.time()
    # args = initial_params('config/ped2_cfg.yml')
    # # 数据集
    # vd = VideoDataLoader(**vars(args))
    # # 模型
    # model = AeMultiOut(input_shape=args.input_shape,
    #                    code_length=args.code_length)
    # # module
    # mdl = MultRecLossModuleFinch(model, **vars(args))
    # # mdl = MultRecLossModule(model, **vars(args))
    #
    # # 使用module训练模型
    # res = trainer_vd_module(args, mdl, vd)
    # end_time = time.time()
    # print(f'running time:{(end_time - stat_time) / 60} m')
if flg =='ave':
    args = initial_params('config/ave_cfg.yml')

    model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length)
    res = train_trainer(args, model)
