import numpy as np
import cv2
import os
import time
import imageio
from tqdm import tqdm
from PIL import Image
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import torchvision
from torchvision.utils import make_grid
from torch import Tensor
from torchvision.transforms.functional import to_tensor

# ----------------------------------------------------------------------------------------------
def savenp2sheet(imgs, savepath, nrow=None):
    """ save multiple imgs (in numpy array type) to a img sheet.
        img sheet is one row.

    imgs: 
        np array of size [N, H, W, 3] or List[array] with array size = [H,W,3] 
    """
    if imgs.ndim == 4:
        img_list = [imgs[i] for i in range(imgs.shape[0])]
        imgs = img_list
    
    imgs_new = []
    for i, img in enumerate(imgs):
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img,(1,2,0))
        
        assert(img.ndim == 3 and img.shape[-1] == 3), img.shape # h,w,3
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs_new.append(img)
    n = len(imgs)
    if nrow is not None:
        n_cols = nrow
    else:
        n_cols=int(n**0.5)
    n_rows=int(np.ceil(n/n_cols))
    print(n_cols)
    print(n_rows)

    imgsheet = cv2.vconcat([cv2.hconcat(imgs_new[i*n_cols:(i+1)*n_cols]) for i in range(n_rows)])
    cv2.imwrite(savepath, imgsheet)
    print(f'saved in {savepath}')

# ----------------------------------------------------------------------------------------------
def save_np_to_img(img, path, norm=True):
    if norm:
        img = (img + 1) / 2 * 255
    img = img.astype(np.uint8)
    image = Image.fromarray(img)
    image.save(path, q=95)

# ----------------------------------------------------------------------------------------------
def npz_to_imgsheet_5d(data_path, res_dir, nrow=None,):
    if isinstance(data_path, str):
        imgs = np.load(data_path)['arr_0'] # NTHWC
    elif isinstance(data_path, np.ndarray):
        imgs = data_path
    else:
        raise Exception
    
    if os.path.isdir(res_dir):
        res_path = os.path.join(res_dir, f'samples.jpg')
    else:
        assert(res_dir.endswith('.jpg'))
        res_path = res_dir
    imgs = np.concatenate([imgs[i] for i in range(imgs.shape[0])], axis=0)
    savenp2sheet(imgs, res_path, nrow=nrow)

# ----------------------------------------------------------------------------------------------
def npz_to_imgsheet_4d(data_path, res_path, nrow=None,):
    if isinstance(data_path, str):
        imgs = np.load(data_path)['arr_0'] # NHWC
    elif isinstance(data_path, np.ndarray):
        imgs = data_path
    else:
        raise Exception
    print(imgs.shape)
    savenp2sheet(imgs, res_path, nrow=nrow)


# ----------------------------------------------------------------------------------------------
def tensor_to_imgsheet(tensor, save_path):
    """ 
        save a batch of videos in one image sheet with shape of [batch_size * num_frames].
        data: [b,c,t,h,w]
    """
    assert(tensor.dim() == 5)
    b,c,t,h,w = tensor.shape
    imgs = [tensor[bi,:,ti, :, :] for bi in range(b) for ti in range(t)]
    torchvision.utils.save_image(imgs, save_path, normalize=True, nrow=t)


# ----------------------------------------------------------------------------------------------
def npz_to_frames(data_path, res_dir, norm, num_frames=None, num_samples=None):
    start = time.time()
    arr = np.load(data_path)
    imgs = arr['arr_0'] # [N, T, H, W, 3]
    print('original data shape: ', imgs.shape)

    if num_samples is not None:
        imgs = imgs[:num_samples, :, :, :, :]
        print('after sample selection: ', imgs.shape)
    
    if num_frames is not None:
        imgs = imgs[:, :num_frames, :, :, :]
        print('after frame selection: ', imgs.shape)

    for vid in tqdm(range(imgs.shape[0]), desc='Video'):
        video_dir = os.path.join(res_dir, f'video{vid:04d}')
        os.makedirs(video_dir, exist_ok=True)
        for fid in range(imgs.shape[1]):
            frame = imgs[vid, fid, :, :, :] #HW3
            save_np_to_img(frame, os.path.join(video_dir, f'frame{fid:04d}.jpg'), norm=norm)
    print('Finish')
    print(f'Total time = {time.time()- start}')

# ----------------------------------------------------------------------------------------------
def npz_to_gifs(data_path, res_dir, duration=0.2, start_idx=0, num_videos=None, mode='gif'):
    os.makedirs(res_dir, exist_ok=True)
    if isinstance(data_path, str):
        imgs = np.load(data_path)['arr_0'] # NTHWC
    elif isinstance(data_path, np.ndarray):
        imgs = data_path
    else:
        raise Exception

    for i in range(imgs.shape[0]):
        frames = [imgs[i,j,:,:,:] for j in range(imgs[i].shape[0])] # [(h,w,3)]
        if mode == 'gif':
            imageio.mimwrite(os.path.join(res_dir, f'samples_{start_idx+i}.gif'), frames, format='GIF', duration=duration)
        elif mode == 'mp4':
            frames = [torch.from_numpy(frame) for frame in frames]
            frames = torch.stack(frames, dim=0).to(torch.uint8) # [T, H, W, C]
            torchvision.io.write_video(os.path.join(res_dir, f'samples_{start_idx+i}.mp4'),
                frames, fps=0.5, video_codec='h264', options={'crf': '10'})
        if i+ 1 == num_videos:
            break

# ----------------------------------------------------------------------------------------------
def fill_with_black_squares(video, desired_len: int) -> Tensor:
    if len(video) >= desired_len:
        return video

    return torch.cat([
        video,
        torch.zeros_like(video[0]).unsqueeze(0).repeat(desired_len - len(video), 1, 1, 1),
    ], dim=0)

# ----------------------------------------------------------------------------------------------
def load_num_videos(data_path, num_videos):
    # data_path can be either data_path of np array 
    if isinstance(data_path, str):
        videos = np.load(data_path)['arr_0'] # NTHWC
    elif isinstance(data_path, np.ndarray):
        videos = data_path
    else:
        raise Exception

    if num_videos is not None:
        videos = videos[:num_videos, :, :, :, :]
    return videos

# ----------------------------------------------------------------------------------------------
def npz_to_video_grid(data_path, out_path, num_frames=None, fps=8, num_videos=None, nrow=None, verbose=True):
    if isinstance(data_path, str):
        videos = load_num_videos(data_path, num_videos)
    elif isinstance(data_path, np.ndarray):
        videos = data_path
    else:
        raise Exception
    n,t,h,w,c = videos.shape

    videos_th = []
    for i in range(n):
        video = videos[i, :,:,:,:]
        images = [video[j, :,:,:] for j in range(t)]
        images = [to_tensor(img) for img in images]
        video = torch.stack(images)
        videos_th.append(video)
    
    if num_frames is None:
        num_frames = videos.shape[1]
    if verbose:
        videos = [fill_with_black_squares(v, num_frames) for v in tqdm(videos_th, desc='Adding empty frames')] # NTCHW
    else:
        videos = [fill_with_black_squares(v, num_frames) for v in videos_th] # NTCHW

    frame_grids = torch.stack(videos).permute(1, 0, 2, 3, 4) # [T, N, C, H, W]
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(n)))
    if verbose:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in tqdm(frame_grids, desc='Making grids')]
    else:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in frame_grids]

    if os.path.dirname(out_path) != "":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frame_grids = (torch.stack(frame_grids) * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
    torchvision.io.write_video(out_path, frame_grids, fps=fps, video_codec='h264', options={'crf': '10'})

# ----------------------------------------------------------------------------------------------
def npz_to_gif_grid(data_path, out_path, n_cols=None, num_videos=20):
    arr = np.load(data_path)
    imgs = arr['arr_0'] # [N, T, H, W, 3]
    imgs = imgs[:num_videos]
    n, t, h, w, c = imgs.shape
    assert(n == num_videos)
    n_cols = n_cols if n_cols else imgs.shape[0]
    n_rows = np.ceil(imgs.shape[0] / n_cols).astype(np.int8)
    H, W = h * n_rows, w * n_cols
    grid = np.zeros((t, H, W, c), dtype=np.uint8)

    for i in range(n_rows):
        for j in range(n_cols):
            if i*n_cols+j < imgs.shape[0]:
                grid[:, i*h:(i+1)*h, j*w:(j+1)*w, :] = imgs[i*n_cols+j, :, :, :, :]
    
    videos = [grid[i] for i in range(grid.shape[0])] # grid: TH'W'C
    imageio.mimwrite(out_path, videos, format='GIF', duration=0.5,palettesize=256)


# ----------------------------------------------------------------------------------------------
def torch_to_video_grid(videos, out_path, num_frames, fps, num_videos=None, nrow=None, verbose=True):
    """
    videos: -1 ~ 1, torch.Tensor, BCTHW
    """
    n,t,h,w,c = videos.shape
    videos_th = [videos[i, ...] for i in range(n)]
    if verbose:
        videos = [fill_with_black_squares(v, num_frames) for v in tqdm(videos_th, desc='Adding empty frames')] # NTCHW
    else:
        videos = [fill_with_black_squares(v, num_frames) for v in videos_th] # NTCHW

    frame_grids = torch.stack(videos).permute(1, 0, 2, 3, 4) # [T, N, C, H, W]
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(n)))
    if verbose:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in tqdm(frame_grids, desc='Making grids')]
    else:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in frame_grids]

    if os.path.dirname(out_path) != "":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frame_grids = ((torch.stack(frame_grids) + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
    torchvision.io.write_video(out_path, frame_grids, fps=fps, video_codec='h264', options={'crf': '10'})
