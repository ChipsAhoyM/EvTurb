import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr



def event_stack_with_polarity(event, H, W, T=32 )->torch.Tensor:
    """
    N * (t, x, y, p) -> (T*2, H, W)
    """


    polarity_maps = np.zeros((T, 2, H, W), dtype=int)

    min_time = event[0, 0]
    max_time = event[-1, 0]
    delta_time = (max_time - min_time) // T

    bin_indices = ((event[:, 0] - min_time) // delta_time).astype(int)

    bin_indices = np.clip(bin_indices, 0, T - 1)

    polarity_values = (event[:, 3] == 1).astype(int)
    np.add.at(polarity_maps, (bin_indices, polarity_values, event[:, 2], event[:, 1]), 1)
    return torch.from_numpy(polarity_maps).reshape(T * 2, H, W)

def event_stack_without_polarity(event, H, W, T=32 )->torch.Tensor:
    """
    N * (t, x, y, p) -> (T*2, H, W)
    """


    polarity_maps = np.zeros((T, H, W), dtype=int)

    min_time = event[0, 0]
    max_time = event[-1, 0]
    delta_time = (max_time - min_time) // T

    bin_indices = ((event[:, 0] - min_time) // delta_time).astype(int)

    bin_indices = np.clip(bin_indices, 0, T - 1)

    polarity_values = event[:, 3]
    np.add.at(polarity_maps, (bin_indices, event[:, 2], event[:, 1]), polarity_values)
    return torch.from_numpy(polarity_maps).reshape(T, H, W)

def random_crop(img:torch.Tensor, size = (512, 512))->torch.Tensor:
    """
    Random crop the image
    """

    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)
    H, W = img.shape[-2:]

    x = np.random.randint(0, W - size[1])
    y = np.random.randint(0, H - size[0])

    return img[..., y:y + size[0], x:x + size[1]]

def compute_psnr(pred, gt):
    """
    Compute the PSNR
    """
    psnr = 0.0
    batch_size = pred.shape[0]
    for i in range(batch_size):
        # print(pred[i].dtype, gt[i].dtype)
        psnr += compare_psnr(pred[i].detach().cpu().numpy(), gt[i].detach().cpu().numpy())
    return psnr / batch_size

def compute_ssim(pred, gt):
    """
    Compute the SSIM
    """
    ssim = 0.0
    batch_size = pred.shape[0]
    for i in range(batch_size):
        ssim += compare_ssim(pred[i].detach().cpu().numpy(), gt[i].detach().cpu().numpy(), channel_axis = 0, data_range = 1)
    return ssim / batch_size


def process_stack(idx, dir_lst):
    i = dir_lst[idx]
    npy_files = os.listdir(i)
    npy_files = [os.path.join(i.split('/')[-1], j) for j in npy_files if j.endswith('.npy')]
    npy_files = sorted(npy_files)
    for j in npy_files:
        event = np.load(os.path.join(dir, j))
        event = event_stack_without_polarity(event, 592, 592, T=32)
        # convert to int8
        event = event.type(torch.int8)
        save_path = os.path.join(save_dir, j.split('/')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_dir, j), event)

def process_var_map(idx, dir_lst):
    i = dir_lst[idx]
    npy_files = os.listdir(i)
    npy_files = [os.path.join(i.split('/')[-1], j) for j in npy_files if j.endswith('.npy')]
    npy_files = sorted(npy_files)
    for j in npy_files:
        event = np.load(os.path.join(dir, j))
        event = event_stack_without_polarity(event, 592, 592, T=1024)
        # convert to int8
        event = event.type(torch.int8)
        # event = event.to(device)
        variance_map = compute_variance_map(event, a0=1, c = 0.2)
        # variance_map = variance_map.cpu().numpy()
        save_path = os.path.join(save_dir, j.split('/')[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_dir, j), variance_map)

def compute_variance_map(event, a0 = 1, c = 0.2):
    '''
    event: torch.Tensor, (T, H, W)
    '''
    # event = event.numpy()
    H, W = event.shape[-2:]
    val_map = torch.cumsum(event, axis = 0)
    val_map = val_map.type(torch.float32)
    # val_map = a0 * torch.exp(c * val_map)
    var_map = torch.var(val_map, axis = 0)
    return (var_map - torch.min(var_map)) / (torch.max(var_map) - torch.min(var_map))


   
if __name__ == '__main__':
    import os
    from tqdm import tqdm
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir = '/openbayes/input/input0/Event'
    save_dir = '/openbayes/input/input0/Var_map'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dir_lst = os.listdir(dir)
    dir_lst = [os.path.join(dir, i) for i in dir_lst]
    dir_lst = sorted(dir_lst)
    # print(dir_lst)
    # import cv2

    from multiprocessing import Pool, cpu_count
    with Pool(cpu_count()) as p:
        p.starmap(process_var_map, [(i, dir_lst) for i in range(len(dir_lst))])
            
        






