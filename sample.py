import argparse
import os

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from trainer import Trainer
from utils import get_config, unloader, get_model_list
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_sample_test', type=int, default=20)
parser.add_argument('--n_test', type=int, default=20)
args = parser.parse_args()

conf_file = os.path.join(args.name, 'configs.yaml')
config = get_config(conf_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transform_list = [transforms.ToTensor(),
                  transforms.Normalize(0.5, 0.5)]
transform = transforms.Compose(transform_list)

if __name__ == '__main__':
    data = np.load(config['data_root'])
    data = data[0:4]
    num = 40

    per = np.random.permutation(data.shape[1])
    data = data[:, per, :, :, :]
    data_for_gen = data[:, :num, :, :, :]

    gen_num = 360

    trainer = Trainer(config)
    if args.ckpt:
        last_model_name = os.path.join(args.name, 'checkpoints', args.ckpt)
    else:
        last_model_name = get_model_list(os.path.join(args.name, 'checkpoints'), "gen")
    trainer.load_ckpt(last_model_name)
    trainer.cuda()
    trainer.eval()
    for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
        for i in range(gen_num):
            idx = np.random.choice(data_for_gen.shape[1], args.n_sample_test)
            imgs = data_for_gen[cls, idx, :, :, :]
            imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(0).cuda()
            fake_x = trainer.generate(imgs)
            output = unloader(fake_x[0].cpu())
            output.save(os.path.join('./results/sdu/sample', str(cls+1), '{}.png'.format(str(i).zfill(3))), 'png')
