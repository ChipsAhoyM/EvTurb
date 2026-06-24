import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from Dataset import EvturbDataset
from UNet import Whole
from utils import compute_psnr, compute_ssim


def get_args():
    parser = argparse.ArgumentParser(description='EvTurb evaluation')
    parser.add_argument('--data_root', type=str, required=True, help='Path to TurbEvent dataset root')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--spatial_offsets', nargs='+', type=int, default=[0],
                        help='Spatial offsets to test robustness (e.g. --spatial_offsets 0 1 2 3)')
    parser.add_argument('--temporal_offsets', nargs='+', type=int, default=[0],
                        help='Temporal offsets to test robustness (e.g. --temporal_offsets 0 1 2 3)')
    return parser.parse_args()


def evaluate(model, val_loader, device, save_dir):
    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for data in tqdm(val_loader):
            img = data['frame'].to(device)
            event = data['event'].to(device)
            gt = data['gt'].to(device)
            vp = data['vp'].to(device)
            filename = data['filename']

            out = model(img, event, vp)
            out = torch.clamp(out, 0, 1)

            val_psnr += compute_psnr(out, gt)
            val_ssim += compute_ssim(out, gt)

            out_img = out.cpu().squeeze()
            sub_dir = os.path.join(save_dir, filename[0].split('/')[-2])
            os.makedirs(sub_dir, exist_ok=True)
            transforms.ToPILImage()(out_img).save(os.path.join(sub_dir, filename[0].split('/')[-1]))

    avg_psnr = val_psnr / len(val_loader)
    avg_ssim = val_ssim / len(val_loader)
    print(f"PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Whole().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model.eval()

    spatial_offset_lst = [(s, s) for s in args.spatial_offsets]
    temporal_offset_lst = args.temporal_offsets

    for spatial_offset in spatial_offset_lst:
        for temporal_offset in temporal_offset_lst:
            result_dir = os.path.join(
                args.save_dir, f'spatial_{spatial_offset[0]}_{spatial_offset[1]}_temporal_{temporal_offset}'
            )
            dataset = EvturbDataset(
                root=args.data_root, mode='val', img_size=(512, 512),
                spatial_offset=spatial_offset, temporal_offset=temporal_offset
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
            print(f"\nEvaluating: spatial_offset={spatial_offset}, temporal_offset={temporal_offset}")
            evaluate(model, loader, device, result_dir)
