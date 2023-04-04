import os
import torch
from PIL import Image

from lvdm.models.modules.lora import net_load_lora
from lvdm.utils.common_utils import instantiate_from_config


# ------------------------------------------------------------------------------------------
def load_model(config, ckpt_path, gpu_id=None, inject_lora=False, lora_scale=1.0, lora_path=''):
    print(f"Loading model from {ckpt_path}")
    
    # load sd
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    try:
        global_step = pl_sd["global_step"]
        epoch = pl_sd["epoch"]
    except:
        global_step = -1
        epoch = -1
    
    # load sd to model
    try:
        sd = pl_sd["state_dict"]
    except:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=True)

    if inject_lora:
        net_load_lora(model, lora_path, alpha=lora_scale)
    
    # move to device & eval
    if gpu_id is not None:
        model.to(f"cuda:{gpu_id}")
    else:
        model.cuda()
    model.eval()

    return model, global_step, epoch


# ------------------------------------------------------------------------------------------
@torch.no_grad()
def get_conditions(prompts, model, batch_size, cond_fps=None,):
    
    if isinstance(prompts, str) or isinstance(prompts, int):
        prompts = [prompts]
    if isinstance(prompts, list):
        if len(prompts) == 1:
            prompts = prompts * batch_size
        elif len(prompts) == batch_size:
            pass
        else:
            raise ValueError(f"invalid prompts length: {len(prompts)}")
    else:
        raise ValueError(f"invalid prompts: {prompts}")
    assert(len(prompts) == batch_size)
    
    # content condition: text / class label
    c = model.get_learned_conditioning(prompts)
    key = 'c_concat' if model.conditioning_key == 'concat' else 'c_crossattn'
    c = {key: [c]}

    # temporal condition: fps
    if getattr(model, 'cond_stage2_config', None) is not None:
        if model.cond_stage2_key == "temporal_context":
            assert(cond_fps is not None)
            batch = {'fps': torch.tensor([cond_fps] * batch_size).long().to(model.device)}
            fps_embd = model.cond_stage2_model(batch)
            c[model.cond_stage2_key] = fps_embd
    
    return c


# ------------------------------------------------------------------------------------------
def make_model_input_shape(model, batch_size, T=None):
    image_size = [model.image_size, model.image_size] if isinstance(model.image_size, int) else model.image_size
    C = model.model.diffusion_model.in_channels
    if T is None:
        T = model.model.diffusion_model.temporal_length
    shape = [batch_size, C, T, *image_size]
    return shape


# ------------------------------------------------------------------------------------------
def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def torch_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    if sample.dim() == 5:
        sample = sample.permute(0, 2, 3, 4, 1)
    else:
        sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def make_sample_dir(opt, global_step=None, epoch=None):
    if not getattr(opt, 'not_automatic_logdir', False):
        gs_str = f"globalstep{global_step:09}" if global_step is not None else "None"
        e_str = f"epoch{epoch:06}" if epoch is not None else "None"
        ckpt_dir = os.path.join(opt.logdir, f"{gs_str}_{e_str}")
        
        # subdir name
        if opt.prompt_file is not None:
            subdir = f"prompts_{os.path.splitext(os.path.basename(opt.prompt_file))[0]}"
        else:
            subdir = f"prompt_{opt.prompt[:10]}"
        subdir += "_DDPM" if opt.vanilla_sample else f"_DDIM{opt.custom_steps}steps"
        subdir += f"_CfgScale{opt.scale}"
        if opt.cond_fps is not None:
            subdir += f"_fps{opt.cond_fps}"
        if opt.seed is not None:
            subdir += f"_seed{opt.seed}"

        return os.path.join(ckpt_dir, subdir)
    else:
        return opt.logdir
