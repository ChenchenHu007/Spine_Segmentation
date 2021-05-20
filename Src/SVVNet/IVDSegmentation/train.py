import os
import sys
import random
import numpy as np

import torch

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

if os.path.abspath('../..') not in sys.path:
    sys.path.insert(0, os.path.abspath('../..'))

import argparse

from utils.tools import csv_to_catalogue
from DataLoader.dataloader_IVDsegmentation import get_loader
from NetworkTrainer.network_trainer import NetworkTrainer
from model import Model
from online_evaluation import online_evaluation
from loss import Loss

if __name__ == '__main__':

    # added by ChenChen Hu
    print('This script has been modified by Chenchen Hu !')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for training (default: 2)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=0,
                        help='list_GPU_ids for training (default: 0)')
    parser.add_argument('--max_iter',  type=int, default=50000,
                        help='training iterations(default: 50000)')
    # added by Chenchen Hu
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--catalogue', type=int, default=0)
    parser.add_argument('--latest', type=int, default=0,
                        help='load the latest model')
    parser.add_argument('--model_path', type=str, default='../../../Output/IVD_Segmentation/latest.pkl')
    parser.add_argument('--model_type', type=str, default='Unet_base',
                        help='the type of Unet(Unet_base or Unet_large)')
    parser.add_argument('--seed', type=int, default=68,
                        help='set the seed')

    args = parser.parse_args()

    torch.manual_seed(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'IVD_Segmentation'
    trainer.setting.output_dir = '../../../Output/IVD_Segmentation'
    list_GPU_ids = args.list_GPU_ids
    csv_path = '../../Catalogue' + '/' + str(args.catalogue) + '.csv'
    catalogue = csv_to_catalogue(csv_path)

    # setting.network is an object
    if args.model_type == 'Unet_base':
        trainer.setting.network = Model(in_ch=2, out_ch=args.num_classes,
                                        list_ch=[-1, 16, 32, 64, 128, 256],)
        print('Loading Unet_base !')
    else:
        trainer.setting.network = Model(in_ch=2, out_ch=args.num_classes,
                                        list_ch=[-1, 32, 64, 128, 256, 512])
        print('Loading Unet_large !')

    trainer.setting.max_iter = args.max_iter

    trainer.setting.train_loader = get_loader(
        catalogue=catalogue,
        batch_size=args.batch_size,  # 2
        num_samples_per_epoch=args.batch_size * 1000,
        num_works=4,
        phase='train'
    )

    trainer.setting.val_loader = get_loader(
        catalogue=catalogue,
        batch_size=1,
        num_samples_per_epoch=20,
        num_works=1,
        phase='val'
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
                             cfgs={
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
