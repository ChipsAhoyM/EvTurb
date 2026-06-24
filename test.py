"""Stage 1 (D-Net) inference: generates deblurred outputs for Stage 2 training."""
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from Dataset import EvturbDataset
from UNet import DeblurNet
from utils import compute_psnr, compute_ssim


def get_args():
    parser = argparse.ArgumentParser(description='EvTurb Stage 1 (D-Net) inference')
    parser.add_argument('--data_root', type=str, required=True, help='Path to TurbEvent dataset root')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to D-Net checkpoint')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save Stage 1 outputs')
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


def test(model, val_loader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for data in tqdm(val_loader):
            img = data['frame'].to(device)
            event = data['event'].to(device)
            filename = data['filename']

            out = model(img, event)
            out = torch.clamp(out, 0, 1)

            out_img = out.cpu().squeeze()
            sub_dir = os.path.join(save_dir, filename[0].split('/')[-2])
            os.makedirs(sub_dir, exist_ok=True)
            transforms.ToPILImage()(out_img).save(os.path.join(sub_dir, filename[0].split('/')[-1]))


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeblurNet().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])

    dataset = EvturbDataset(root=args.data_root, mode='val', img_size=(512, 512))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    test(model, loader, device, args.save_dir)
