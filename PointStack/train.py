import os
import torch
import argparse
import datetime
import numpy as np
import random
import shutil
import logging

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.builders import build_dataset, build_network, build_optimizer
from utils.runtime_utils import cfg, cfg_from_yaml_file, validate
from utils.vis_utils import visualize_numpy


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/partnormal/pointstack.yaml',
                        help='specify the config for training')
    parser.add_argument('--exp_name', type=str, default='experiments',
                        help='specify experiment name for saving outputs')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed number')
    parser.add_argument('--val_steps', type=int, default=1, help='perform validation every n steps')
    parser.add_argument('--pretrained_ckpt', type=str, default='experiments/PartNormal/experiments/ckpt/ckpt-best.pth',
                        help='path to pretrained ckpt')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    exp_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy2(args.cfg_file, exp_dir)

    return args, cfg


args, cfg = parse_config()

random_seed = cfg.RANDOM_SEED  # Setup seed for reproducibility
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# log
log_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / 'experiments' / 'log'
log_dir.mkdir(exist_ok=True)
logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('%s/PointStack.txt' % (log_dir))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log_string(str):
    logger.info(str)
    print(str)


# Build Dataloader
train_dataset = build_dataset(cfg, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=cfg.OPTIMIZER.BATCH_SIZE, shuffle=True, drop_last=True)

val_dataset = build_dataset(cfg, split='val')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

# Build Network and Optimizer
net = build_network(cfg)


opt, scheduler = build_optimizer(cfg, net.parameters(), len(train_dataloader))

try:
    checkpoint = torch.load(args.pretrained_ckpt)
    pretrained_state_dict = checkpoint['model_state_dict']
    first_epoch = checkpoint['epoch']
    max_acc = checkpoint['best Val mIoU']
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    for k, v in net.state_dict().items():
        if (v.shape != pretrained_state_dict[k].shape):
            del pretrained_state_dict[k]

    net.load_state_dict(pretrained_state_dict, strict=False)
    log_string('Use Pretrained Model')
except:
    log_string('No exsiting model, starting training from scratch...')
    first_epoch = 1
    max_acc = 0

net = net.cuda()


from torch.utils.tensorboard import SummaryWriter

ckpt_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'ckpt'
tensorboard_dir = cfg.ROOT_DIR / 'experiments' / cfg.DATASET.NAME / args.exp_name / 'tensorboard'

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

writer = SummaryWriter(tensorboard_dir)

min_loss = 1e20

steps_cnt = 0
epoch_cnt = 0


for epoch in tqdm(range(first_epoch, cfg.OPTIMIZER.MAX_EPOCH + 1)):
    opt.zero_grad()
    net.zero_grad()
    net.train()
    loss = 0
    for data_dic in tqdm(train_dataloader):
        data_dic = net(data_dic)
        loss, loss_dict = net.get_loss(data_dic, smoothing=True, is_segmentation=cfg.DATASET.IS_SEGMENTATION)
        loss = loss
        loss.backward()
        steps_cnt += 1

        # if (steps_cnt)%(cfg.OPTIMIZER.GRAD_ACCUMULATION) == 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.OPTIMIZER.GRAD_CLIP)
        opt.step()
        opt.zero_grad()
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        writer.add_scalar('steps/loss', loss, steps_cnt)
        writer.add_scalar('steps/lr', lr, steps_cnt)

        for k, v in loss_dict.items():
            writer.add_scalar('steps/loss_' + k, v, steps_cnt)

    if (epoch % args.val_steps) == 0:
        val_dict = validate(net, val_dataloader, net.get_loss, 'cuda', is_segmentation=cfg.DATASET.IS_SEGMENTATION)

        log_string('===========================Epoch %d ======================' % epoch)

        if cfg.DATASET.IS_SEGMENTATION:
            writer.add_scalar('epochs/val_miou', val_dict['miou'], epoch)
            log_string('Val mIoU: %.3f' % val_dict['miou'])

        else:
            writer.add_scalar('epochs/val_loss', val_dict['loss'], epoch_cnt)
            writer.add_scalar('epochs/val_acc', val_dict['acc'], epoch_cnt)
            writer.add_scalar('epochs/val_acc_avg', val_dict['acc_avg'], epoch_cnt)
            print('Val Loss: ', val_dict['loss'], 'Val Accuracy: ', val_dict['acc'], 'Val Avg Accuracy: ',
                  val_dict['acc_avg'])

            for k, v in val_dict['loss_dic'].items():
                writer.add_scalar('epochs/val_loss_' + k, v, epoch_cnt)

        if cfg.DATASET.IS_SEGMENTATION:
            if val_dict['miou'] > max_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best Val mIoU': val_dict['miou'],
                    'scheduler': scheduler.state_dict()
                }, ckpt_dir / 'ckpt-best.pth')
                log_string('best Val mIoU: %.3f' % val_dict['miou'])

                max_acc = val_dict['miou']
            else:
                print('best Val mIoU: ', max_acc)
        else:
            if val_dict['acc'] > max_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': val_dict['loss'],
                }, ckpt_dir / 'ckpt-best.pth')

                max_acc = val_dict['acc']

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, ckpt_dir / 'ckpt-last.pth')
