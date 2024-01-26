import os
import time
from omegaconf import OmegaConf
import torch
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling
from utils.utils import instantiate_from_config
from huggingface_hub import hf_hub_download

class Text2Video():
    def __init__(self,result_dir='./tmp/',gpu_num=1) -> None:
        self.download_model()
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        ckpt_path='checkpoints/base_512_v2/model.ckpt'
        config_file='configs/inference_t2v_512_v2.0.yaml'
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint']=False   
        model_list = []
        for gpu_id in range(gpu_num):
            model = instantiate_from_config(model_config)
            # model = model.cuda(gpu_id)
            assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
            model = load_model_checkpoint(model, ckpt_path)
            model.eval()
            model_list.append(model)
        self.model_list = model_list
        self.save_fps = 8

    def get_prompt(self, prompt, steps=50, cfg_scale=12.0, eta=1.0, fps=16):
        torch.cuda.empty_cache()
        print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        start = time.time()
        gpu_id=0
        if steps > 60:
            steps = 60 
        model = self.model_list[gpu_id]
        model = model.cuda()
        batch_size=1
        channels = model.model.diffusion_model.in_channels
        frames = model.temporal_length
        h, w = 320 // 8, 512 // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # text cond
        text_emb = model.get_learned_conditioning([prompt])
        cond = {"c_crossattn": [text_emb], "fps": fps}
        
        ## inference
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)
        ## b,samples,c,t,h,w
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
        prompt_str=prompt_str[:30]

        save_videos(batch_samples, self.result_dir, filenames=[prompt_str], fps=self.save_fps)
        print(f"Saved in {prompt_str}. Time used: {(time.time() - start):.2f} seconds")
        model=model.cpu()
        return os.path.join(self.result_dir, f"{prompt_str}.mp4")
    
    def download_model(self):
        REPO_ID = 'VideoCrafter/VideoCrafter2'
        filename_list = ['model.ckpt']
        if not os.path.exists('./checkpoints/base_512_v2/'):
            os.makedirs('./checkpoints/base_512_v2/')
        for filename in filename_list:
            local_file = os.path.join('./checkpoints/base_512_v2/', filename)

            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/base_512_v2/', local_dir_use_symlinks=False)

    
if __name__ == '__main__':
    t2v = Text2Video()
    video_path = t2v.get_prompt('a black swan swims on the pond')
    print('done', video_path)