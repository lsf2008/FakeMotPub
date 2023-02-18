from aemodel.ae_multi_out import AeMultiOut
from config.param_seting import initial_params
import trainer

args = initial_params('../config/ped2_cfg.yml')

model = AeMultiOut(input_shape=args.crop_size,
                   code_length=args.code_length)

res = trainer.train_trainer(args, model)
