import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf

import torch
from decord import VideoReader, cpu
import torchvision
from pytorch_lightning import seed_everything

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import instantiate_from_config
from lvdm.utils.saving_utils import tensor_to_mp4


def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt, adapter_ckpt=None):
    print('>>> Loading checkpoints ...')
    if adapter_ckpt:
        ## main model
        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print('@model checkpoint loaded.')
        ## adapter
        state_dict = torch.load(adapter_ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.adapter.load_state_dict(state_dict, strict=True)
        print('@adapter checkpoint loaded.')
    else:
        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print('@model checkpoint loaded.')
    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def load_video(filepath, frame_stride, video_size=(256,256), video_frames=16):
    vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
    max_frames = len(vidreader)
    temp_stride = max_frames // video_frames if frame_stride == -1 else frame_stride
    if temp_stride * (video_frames-1) >= max_frames:
        print(f'Warning: default frame stride is used because the input video clip {max_frames} is not long enough.')
        temp_stride = max_frames // video_frames
    frame_indices = [temp_stride*i for i in range(video_frames)]
    frames = vidreader.get_batch(frame_indices)
        
    ## [t,h,w,c] -> [c,t,h,w]
    frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
    frame_tensor = (frame_tensor / 255. - 0.5) * 2    
    return frame_tensor


def save_results(prompt, samples, inputs, filename, realdir, fakedir, fps=10):
    ## save prompt
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    path = os.path.join(realdir, "%s.txt"%filename)
    with open(path, 'w') as f:
        f.write(f'{prompt}')
        f.close()

    ## save video
    videos = [inputs, samples]
    savedirs = [realdir, fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], "%s.mp4"%filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})


def adapter_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, unconditional_guidance_scale_temporal=None, **kwargs):
    ddim_sampler = DDIMSampler(model)

    batch_size = noise_shape[0]
    ## get condition embeddings (support single prompt only)
    if isinstance(prompts, str):
        prompts = [prompts]
    cond = model.get_learned_conditioning(prompts)
    if unconditional_guidance_scale != 1.0:
        prompts = batch_size * [""]
        uc = model.get_learned_conditioning(prompts)
    else:
        uc = None
    
    ## adapter features: process in 2D manner
    b, c, t, h, w = videos.shape
    extra_cond = model.get_batch_depth(videos, (h,w))
    features_adapter = model.get_adapter_features(extra_cond)

    batch_variants = []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
                                            features_adapter=features_adapter,
                                            **kwargs
                                            )        
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples, decode_bs=1, return_cpu=False)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5), extra_cond


def run_inference(args, gpu_idx):
    ## model config
    config = OmegaConf.load(args.base)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_idx)
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path, args.adapter_ckpt)
    model.eval()

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.channels
    frames = model.temporal_length
    noise_shape = [args.bs, channels, args.num_frames, h, w]
    
    ## inference
    start = time.time()
    prompt = args.prompt
    video = load_video(args.video, args.frame_stride, video_size=(args.height, args.width), video_frames=args.num_frames)
    video = video.unsqueeze(0).to("cuda")
    with torch.no_grad():
        batch_samples, batch_conds = adapter_guided_synthesis(model, prompt, video, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                                args.unconditional_guidance_scale, args.unconditional_guidance_scale_temporal)
    batch_samples = batch_samples[0]
    os.makedirs(args.savedir, exist_ok=True)
    filename = f"{args.prompt}_seed{args.seed}"
    filename = filename.replace("/", "_slash_") if "/" in filename else filename
    filename = filename.replace(" ", "_") if " " in filename else filename
    tensor_to_mp4(video=batch_conds.detach().cpu(), savepath=os.path.join(args.savedir, f'{filename}_depth.mp4'), fps=10)
    tensor_to_mp4(video=batch_samples.detach().cpu(), savepath=os.path.join(args.savedir, f'{filename}_sample.mp4'), fps=10)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--adapter_ckpt", type=str, default=None, help="adapter checkpoint path")
    parser.add_argument("--base", type=str, help="config (yaml) path")
    parser.add_argument("--prompt", type=str, default=None, help="prompt string")
    parser.add_argument("--video", type=str, default=None, help="video path")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=-1, help="frame extracting from input video")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    parser.add_argument("--seed", type=int, default=2023, help="seed for seed_everything")
    parser.add_argument("--num_frames", type=int, default=16, help="number of input frames")    
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoVideoGen cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    rank = 0
    run_inference(args, rank)