# -*- encoding: utf-8 -*-
# -*- encoding: utf-8 -*-
import os
import sys
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

import argparse

from ..DataLoader.dataloader_3D import get_loader
from ..NetworkTrainer.network_trainer import NetworkTrainer
from model import Model
from online_evaluation import online_evaluation
from loss import Loss

if __name__ == '__main__':

    # added by ChenChen Hu
    print('This script has been modified by Chenchen Hu !')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for training (default: 2)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[1, 0],
                        help='list_GPU_ids for training (default: [1, 0])')
    parser.add_argument('--max_iter',  type=int, default=80000,
                        help='training iterations(default: 80000)')
    # added by Chenchen Hu
    parser.add_argument('--latest', type=int, default=0,
                        help='load the latest model')
    parser.add_argument('--model_path', type=str, default='../../Output/SpineSegmentationC3D/latest.pkl')

    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'Spine_Segmentation_C3D'
    trainer.setting.output_dir = '../../Output/Spine_Segmentation_C3D'
    list_GPU_ids = args.list_GPU_ids

    # setting.network is an object
    trainer.setting.network = Model(in_ch=9, out_ch=1,
                                    list_ch_A=[-1, 16, 32, 64, 128, 256],
                                    list_ch_B=[-1, 32, 64, 128, 256, 512])  # list_ch_B=[-1, 32, 64, 128, 256, 512]

    trainer.setting.max_iter = args.max_iter  # 80000 or 100000

    trainer.setting.train_loader, trainer.setting.val_loader = get_loader(  # -> data.DataLoader
        train_bs=args.batch_size,  # 2
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch => 1000 samples per epoch
        val_num_samples_per_epoch=1,
        num_works=4
    )

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr': 3e-4,
                              'weight_decay': 1e-4
                          }
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='cosine',
                             args={
                                 'T_max': args.max_iter,
                                 'eta_min': 1e-7,
                                 'last_epoch': -1
                             }
                             )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    trainer.set_GPU_device(list_GPU_ids)

    # added by Chenchen Hu
    # load the latest model when the recovery is True and the model exists.
    if args.latest and os.path.exists(args.model_path):
        trainer.init_trainer(ckpt_file=args.model_path,
                             list_GPU_ids=list_GPU_ids,
                             only_network=False)

    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')
