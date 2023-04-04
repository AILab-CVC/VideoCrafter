import os
import time
import argparse
import yaml, math
from tqdm import trange
import torch
import numpy as np
from omegaconf import OmegaConf
import torch.distributed as dist
from pytorch_lightning import seed_everything

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import str2bool
from lvdm.utils.dist_utils import setup_dist, gather_data
from lvdm.utils.saving_utils import npz_to_video_grid, npz_to_imgsheet_5d
from scripts.sample_utils import load_model, get_conditions, make_model_input_shape, torch_to_np


# ------------------------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--ckpt_path", type=str, help="model checkpoint path")
    parser.add_argument("--config_path", type=str, help="model config path (a yaml file)")
    parser.add_argument("--prompt", type=str, help="input text prompts for text2video (a sentence OR a txt file).")
    parser.add_argument("--save_dir", type=str, help="results saving dir", default="results/")
    # device args
    parser.add_argument("--ddp", action='store_true', help="whether use pytorch ddp mode for parallel sampling (recommend for multi-gpu case)", default=False)
    parser.add_argument("--local_rank", type=int, help="is used for pytorch ddp mode", default=0)
    parser.add_argument("--gpu_id", type=int, help="choose a specific gpu", default=0)
    # sampling args
    parser.add_argument("--n_samples", type=int, help="how many samples for each text prompt", default=2)
    parser.add_argument("--batch_size", type=int, help="video batch size for sampling", default=1)
    parser.add_argument("--decode_frame_bs", type=int, help="frame batch size for framewise decoding", default=1)
    parser.add_argument("--sample_type", type=str, help="ddpm or ddim", default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, help="ddim sampling -- number of ddim denoising timesteps", default=50)
    parser.add_argument("--eta", type=float, help="ddim sampling -- eta (0.0 yields deterministic sampling, 1.0 yields random sampling)", default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=15.0, help="classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="fix a seed for randomness (If you want to reproduce the sample results)")
    parser.add_argument("--show_denoising_progress", action='store_true', default=False, help="whether show denoising progress during sampling one batch",)
    # lora args
    parser.add_argument("--lora_path", type=str, help="lora checkpoint path")
    parser.add_argument("--inject_lora", action='store_true', default=False, help="",)
    parser.add_argument("--lora_scale", type=float, default=None, help="scale for lora weight")
    parser.add_argument("--lora_trigger_word", type=str, default="", help="",)
    # saving args
    parser.add_argument("--save_mp4", type=str2bool, default=True, help="whether save samples in separate mp4 files", choices=["True", "true", "False", "false"])
    parser.add_argument("--save_mp4_sheet", action='store_true', default=False, help="whether save samples in mp4 file",)
    parser.add_argument("--save_npz", action='store_true', default=False, help="whether save samples in npz file",)
    parser.add_argument("--save_jpg", action='store_true', default=False, help="whether save samples in jpg file",)
    parser.add_argument("--save_fps", type=int, default=8, help="fps of saved mp4 videos",)
    return parser

# ------------------------------------------------------------------------------------------
def sample_denoising_batch(model, noise_shape, condition, *args,
                           sample_type="ddim", sampler=None, 
                           ddim_steps=None, eta=None,
                           unconditional_guidance_scale=1.0, uc=None,
                           denoising_progress=False,
                           **kwargs,
                           ):
    
    if sample_type == "ddpm":
        samples = model.p_sample_loop(cond=condition, shape=noise_shape,
                                      return_intermediates=False, 
                                      verbose=denoising_progress,
                                      **kwargs,
                                      )
    elif sample_type == "ddim":
        assert(sampler is not None)
        assert(ddim_steps is not None)
        assert(eta is not None)
        ddim_sampler = sampler
        samples, _ = ddim_sampler.sample(S=ddim_steps,
                                         conditioning=condition,
                                         batch_size=noise_shape[0],
                                         shape=noise_shape[1:],
                                         verbose=denoising_progress,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=uc,
                                         eta=eta,
                                         **kwargs,
                                        )
    else:
        raise ValueError
    return samples


# ------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_text2video(model, prompt, n_samples, batch_size,
                      sample_type="ddim", sampler=None, 
                      ddim_steps=50, eta=1.0, cfg_scale=7.5, 
                      decode_frame_bs=1,
                      ddp=False, all_gather=True, 
                      batch_progress=True, show_denoising_progress=False,
                      ):
    # get cond vector
    assert(model.cond_stage_model is not None)
    cond_embd = get_conditions(prompt, model, batch_size)
    uncond_embd = get_conditions("", model, batch_size) if cfg_scale != 1.0 else None

    # sample batches
    all_videos = []
    n_iter = math.ceil(n_samples / batch_size)
    iterator  = trange(n_iter, desc="Sampling Batches (text-to-video)") if batch_progress else range(n_iter)
    for _ in iterator:
        noise_shape = make_model_input_shape(model, batch_size)
        samples_latent = sample_denoising_batch(model, noise_shape, cond_embd,
                                            sample_type=sample_type,
                                            sampler=sampler,
                                            ddim_steps=ddim_steps,
                                            eta=eta,
                                            unconditional_guidance_scale=cfg_scale, 
                                            uc=uncond_embd,
                                            denoising_progress=show_denoising_progress,
                                            )
        samples = model.decode_first_stage(samples_latent, decode_bs=decode_frame_bs, return_cpu=False)
        
        # gather samples from multiple gpus
        if ddp and all_gather:
            data_list = gather_data(samples, return_np=False)
            all_videos.extend([torch_to_np(data) for data in data_list])
        else:
            all_videos.append(torch_to_np(samples))
    
    all_videos = np.concatenate(all_videos, axis=0)
    assert(all_videos.shape[0] >= n_samples)
    return all_videos


# ------------------------------------------------------------------------------------------
def save_results(videos, save_dir, 
                 save_name="results", save_fps=8, save_mp4=True, 
                 save_npz=False, save_mp4_sheet=False, save_jpg=False
                 ):
    if save_mp4:
        save_subdir = os.path.join(save_dir, "videos")
        os.makedirs(save_subdir, exist_ok=True)
        for i in range(videos.shape[0]):
            npz_to_video_grid(videos[i:i+1,...], 
                              os.path.join(save_subdir, f"{save_name}_{i:03d}.mp4"), 
                              fps=save_fps)
        print(f'Successfully saved videos in {save_subdir}')
    
    if save_npz:
        save_path = os.path.join(save_dir, f"{save_name}.npz")
        np.savez(save_path, videos)
        print(f'Successfully saved npz in {save_path}')
    
    if save_mp4_sheet:
        save_path = os.path.join(save_dir, f"{save_name}.mp4")
        npz_to_video_grid(videos, save_path, fps=save_fps)
        print(f'Successfully saved mp4 sheet in {save_path}')

    if save_jpg:
        save_path = os.path.join(save_dir, f"{save_name}.jpg")
        npz_to_imgsheet_5d(videos, save_path, nrow=videos.shape[1])
        print(f'Successfully saved jpg sheet in {save_path}')


# ------------------------------------------------------------------------------------------
def main():
    """
    text-to-video generation
    """
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    os.makedirs(opt.save_dir, exist_ok=True)
    
    # set device
    if opt.ddp:
        setup_dist(opt.local_rank)
        opt.n_samples = math.ceil(opt.n_samples / dist.get_world_size())
        gpu_id = None
    else:
        gpu_id = opt.gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    
    # set random seed
    if opt.seed is not None:
        if opt.ddp:
            seed = opt.local_rank + opt.seed
        else:
            seed = opt.seed
        seed_everything(seed)

    # dump args
    fpath = os.path.join(opt.save_dir, "sampling_args.yaml")
    with open(fpath, 'w') as f:
        yaml.dump(vars(opt), f, default_flow_style=False)

    # load & merge config
    config = OmegaConf.load(opt.config_path)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    print("config: \n", config)

    # get model & sampler
    model, _, _ = load_model(config, opt.ckpt_path, 
                             inject_lora=opt.inject_lora, 
                             lora_scale=opt.lora_scale, 
                             lora_path=opt.lora_path
                             )
    ddim_sampler = DDIMSampler(model) if opt.sample_type == "ddim" else None

    # prepare prompt
    if opt.prompt.endswith(".txt"):
        opt.prompt_file = opt.prompt
        opt.prompt = None
    else:
        opt.prompt_file = None

    if opt.prompt_file is not None:
        f = open(opt.prompt_file, 'r')
        prompts, line_idx = [], []
        for idx, line in enumerate(f.readlines()):
            l = line.strip()
            if len(l) != 0:
                prompts.append(l)
                line_idx.append(idx)
        f.close()
        cmd = f"cp {opt.prompt_file} {opt.save_dir}"
        os.system(cmd)
    else:
        prompts = [opt.prompt]
        line_idx = [None]

    if opt.inject_lora:
        assert(opt.lora_trigger_word != '')
        prompts = [p + opt.lora_trigger_word for p in prompts]
    
    # go
    start = time.time()  
    for prompt in prompts:
        # sample
        samples = sample_text2video(model, prompt, opt.n_samples, opt.batch_size,
                          sample_type=opt.sample_type, sampler=ddim_sampler,
                          ddim_steps=opt.ddim_steps, eta=opt.eta, 
                          cfg_scale=opt.cfg_scale,
                          decode_frame_bs=opt.decode_frame_bs,
                          ddp=opt.ddp, show_denoising_progress=opt.show_denoising_progress,
                          )
        # save
        if (opt.ddp and dist.get_rank() == 0) or (not opt.ddp):
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            save_name = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            if opt.seed is not None:
                save_name = save_name + f"_seed{seed:05d}"
            save_results(samples, opt.save_dir, save_name=save_name, save_fps=opt.save_fps)
    print("Finish sampling!")
    print(f"Run time = {(time.time() - start):.2f} seconds")

    if opt.ddp:
        dist.destroy_process_group()


# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()