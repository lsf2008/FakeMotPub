from aemodel.ae_multi_out_wghts import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import train_trainer,trainer_vd_module
# from aemodel.autoencoder import convAE
from dataset.video_dataloader import VideoDataLoader
from aemodel.ae_mlp import Ae1Mlp2, Ae2Mlps, AeMlp, Ae1Mlp3,Ae3Mlps
import prettytable as pt
import time
import utils as utils
import torch
from trainer.fakeMot_cross_module import FakeMotCrossModule
import pytorch_lightning as pl

flg ='auc'
if flg == 'train':

    pl.seed_everything(999999)
    tbl = pt.PrettyTable()
    tbl.field_names = ['auc', 'cmb_coef', 'epoch', 'dataset']

    stat_time = time.time()

    trn_pth = 'E:/dataset/shanghaitech/label/train/'
    trnList = utils.getImgListFrm1Flder(trn_pth, type='*.csv')

    tstPth = 'E:/dataset/shanghaitech/label/test/'
    aucList = []
    for trn in trnList:
        args = initial_params('config/dtsht_cfg.yml')
        tstCSV = tstPth + 'test' + str(trn).split('_')[-1]

        args.train_dataPath = str(trn)
        args.val_dataPath = tstCSV

        # 数据集
        vd = VideoDataLoader(**vars(args))

        # 模型
        model = Ae1Mlp2(input_shape=args.input_shape,
                        code_length=args.code_length)
        # trainer module
        mdl = FakeMotCrossModule(model, **vars(args))

        # 使用module训练模型
        res = trainer_vd_module(args, mdl, vd)

        # print(str(trn)+res['maxAuc'])
        aucList.append(res['maxAuc'])
        sn = 'data/shg/' + str(trn).split('_')[-1].split('.')[0] + '.pt'
        tbl.add_row([res['maxAuc'], res['coef'], res['epoch'], str(trn).split('_')[-1].split('.')[0]])
        torch.save(res, sn)

    end_time = time.time()

    print(tbl)

    print(f'running time:{(end_time - stat_time) / 60} m')
    sv_name = 'data/shg/fakemot_shg.txt'
    with open(sv_name, 'w') as f:
        f.write(tbl.get_string())
    print(f'auc mean: {torch.mean(torch.tensor(aucList))}')

if flg=='auc':
    from pathlib import Path
    obj = Path('data/shg')
    aucList = []
    for f in obj.glob('*.pt'):
        dt = torch.load(str(f))
        aucList.append(dt['maxAuc'])

    print(torch.mean(torch.tensor(aucList)))

