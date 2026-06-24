"""Stage 2 (T-Net) inference: applies tilt correction on Stage 1 outputs."""
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from Dataset import EvturbDataset
from flow import IFNet
from utils import compute_psnr, compute_ssim


def get_args():
    parser = argparse.ArgumentParser(description='EvTurb Stage 2 (T-Net) inference')
    parser.add_argument('--data_root', type=str, required=True, help='Path to TurbEvent dataset root')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to T-Net checkpoint')
    parser.add_argument('--save_dir', type=str, default='./results_stage2')
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


def test(model, val_loader, device, save_dir):
    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for data in tqdm(val_loader):
            img = data['frame'].to(device)
            gt = data['gt'].to(device)
            vp = data['vp'].to(device)
            filename = data['filename']

            out = model(img, vp)
            out = torch.clamp(out, 0, 1)
            val_psnr += compute_psnr(out, gt)
            val_ssim += compute_ssim(out, gt)

            out_img = out.cpu().squeeze()
            sub_dir = os.path.join(save_dir, filename[0].split('/')[-2])
            os.makedirs(sub_dir, exist_ok=True)
            transforms.ToPILImage()(out_img).save(os.path.join(sub_dir, filename[0].split('/')[-1]))

    print(f"PSNR: {val_psnr / len(val_loader):.3f}, SSIM: {val_ssim / len(val_loader):.4f}")


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IFNet().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])

    # second_stage=True loads Stage 1 outputs from data_root/first_stage/
    dataset = EvturbDataset(root=args.data_root, mode='val', img_size=(512, 512), second_stage=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    test(model, loader, device, args.save_dir)
