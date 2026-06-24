"""Two-stage training: Stage 2 (T-Net) with frozen Stage 1 (D-Net).
Run test.py first to generate Stage 1 outputs, then use train-end2end.py for joint training.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import tensorboardX
from tqdm import tqdm
import lpips

from Dataset import EvturbDataset
from UNet import DeblurNet
from flow import IFNet
from utils import compute_psnr, compute_ssim


def get_args():
    parser = argparse.ArgumentParser(description='EvTurb Stage 2 (T-Net) training')
    parser.add_argument('--data_root', type=str, required=True, help='Path to TurbEvent dataset root')
    parser.add_argument('--dnet_checkpoint', type=str, required=True, help='Path to pretrained D-Net checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


def perceptual_loss(x, y, per_loss):
    x = (x - 0.5) * 2
    y = (y - 0.5) * 2
    return per_loss(x, y).mean()


def train(dnet, tnet, train_loader, optimizer, epoch, args, device, per_loss, writer):
    tnet.train()
    criterion_mse = nn.MSELoss()
    running_loss = 0.0
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)):
        img = data['frame'].to(device)
        event = data['event'].to(device)
        gt = data['gt'].to(device)
        vp = data['vp'].to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            out1 = dnet(img, event)
        out = tnet(out1, vp)
        mse_loss = criterion_mse(out, gt)
        perc_loss = perceptual_loss(out, gt, per_loss)
        total_loss = mse_loss + 0.02 * perc_loss

        writer.add_scalar('Train/Loss', total_loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Train/MSE_Loss', mse_loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Train/Perc_Loss', perc_loss.item(), epoch * len(train_loader) + batch_idx)

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Train Loss: {avg_loss:.6f}")
    return avg_loss


def validate(dnet, tnet, val_loader, epoch, args, device, per_loss, writer):
    dnet.eval()
    tnet.eval()
    criterion_mse = nn.MSELoss()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)):
            img = data['frame'].to(device)
            event = data['event'].to(device)
            gt = data['gt'].to(device)
            vp = data['vp'].to(device)

            out1 = dnet(img, event)
            out = tnet(out1, vp)
            out = torch.clamp(out, 0, 1)
            total_loss = criterion_mse(out, gt) + 0.02 * perceptual_loss(out, gt, per_loss)
            val_loss += total_loss.item()
            val_psnr += compute_psnr(out, gt)
            val_ssim += compute_ssim(out, gt)

            if batch_idx % 100 == 0:
                out_imgs = torch.zeros((out.shape[1], out.shape[2], out.shape[3] * args.batch_size_val))
                gt_imgs = torch.zeros((gt.shape[1], gt.shape[2], gt.shape[3] * args.batch_size_val))
                for i in range(min(args.batch_size_val, out.shape[0])):
                    out_imgs[:, :, i * out.shape[3]:(i + 1) * out.shape[3]] = out[i].squeeze()
                    gt_imgs[:, :, i * gt.shape[3]:(i + 1) * gt.shape[3]] = gt[i].squeeze()
                writer.add_image('Validation/Output', out_imgs, epoch * len(val_loader) + batch_idx)
                writer.add_image('Validation/GT', gt_imgs, epoch * len(val_loader) + batch_idx)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_loader)
    avg_val_ssim = val_ssim / len(val_loader)
    writer.add_scalar('Validation/Avg_Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation/Avg_PSNR', avg_val_psnr, epoch)
    writer.add_scalar('Validation/Avg_SSIM', avg_val_ssim, epoch)
    print(f"Epoch {epoch+1} Avg. Val Loss: {avg_val_loss:.6f}, Avg. PSNR: {avg_val_psnr:.3f}, Avg. SSIM: {avg_val_ssim:.3f}")
    return avg_val_loss


def save_model(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)
    print(f"Model saved at epoch {epoch}: {path}")


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    per_loss = lpips.LPIPS(net='vgg').to(device)

    train_dataset = EvturbDataset(root=args.data_root, mode='train', img_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset = EvturbDataset(root=args.data_root, mode='val', img_size=(512, 512))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=args.num_workers)

    dnet = DeblurNet()
    if torch.cuda.device_count() > 1:
        dnet = nn.DataParallel(dnet)
    dnet = dnet.to(device)
    dnet.load_state_dict(torch.load(args.dnet_checkpoint)['model_state_dict'])
    dnet.eval()

    tnet = IFNet()
    if torch.cuda.device_count() > 1:
        tnet = nn.DataParallel(tnet)
    tnet = tnet.to(device)

    optimizer = optim.Adam(tnet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = tensorboardX.SummaryWriter()

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train(dnet, tnet, train_loader, optimizer, epoch, args, device, per_loss, writer)
        val_loss = validate(dnet, tnet, val_loader, epoch, args, device, per_loss, writer)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(tnet, optimizer, scheduler, epoch + 1,
                       os.path.join(args.checkpoint_dir, 'best_model.pth'))

        if (epoch + 1) % 5 == 0:
            save_model(tnet, optimizer, scheduler, epoch + 1,
                       os.path.join(args.checkpoint_dir, f'epoch_{epoch + 1}_model.pth'))

    writer.close()
