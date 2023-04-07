import os
import sys
import gradio as gr
from omegaconf import OmegaConf

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.saving_utils import npz_to_video_grid
from scripts.sample_text2video import sample_text2video
from scripts.sample_utils import load_model
from lvdm.models.modules.lora import change_lora_v2

from huggingface_hub import hf_hub_download


def save_results(videos, save_dir, 
                 save_name="results", save_fps=8
                 ):
    save_subdir = os.path.join(save_dir, "videos")
    os.makedirs(save_subdir, exist_ok=True)
    for i in range(videos.shape[0]):
        npz_to_video_grid(videos[i:i+1,...], 
                            os.path.join(save_subdir, f"{save_name}_{i:03d}.mp4"), 
                            fps=save_fps)
    print(f'Successfully saved videos in {save_subdir}')
    video_path_list = [os.path.join(save_subdir, f"{save_name}_{i:03d}.mp4") for i in range(videos.shape[0])]
    return video_path_list
    

class Text2Video():
    def __init__(self,result_dir='./tmp/') -> None:
        self.download_model()
        config_file = 'models/base_t2v/model_config.yaml'
        ckpt_path = 'models/base_t2v/model.ckpt'
        config = OmegaConf.load(config_file)
        self.lora_path_list = ['','models/videolora/lora_001_Loving_Vincent_style.ckpt',
                                'models/videolora/lora_002_frozenmovie_style.ckpt',
                                'models/videolora/lora_003_MakotoShinkaiYourName_style.ckpt',
                                'models/videolora/lora_004_coco_style.ckpt']
        self.lora_trigger_word_list = ['','Loving Vincent style', 'frozenmovie style', 'MakotoShinkaiYourName style', 'coco style']
        model, _, _ = load_model(config, ckpt_path, gpu_id=0, inject_lora=False)
        self.model = model
        self.last_time_lora = ''
        self.last_time_lora_scale = 1.0
        self.result_dir = result_dir
        self.save_fps = 8
        self.ddim_sampler = DDIMSampler(model) 
        self.origin_weight = None

    def get_prompt(self, input_text, steps=50, model_index=0, eta=1.0, cfg_scale=15.0, lora_scale=1.0):
        if model_index > 0:
            input_text = input_text + ', ' + self.lora_trigger_word_list[model_index]
        inject_lora = model_index > 0
        self.origin_weight = change_lora_v2(self.model, inject_lora=inject_lora, lora_scale=lora_scale, lora_path=self.lora_path_list[model_index],
                    last_time_lora=self.last_time_lora, last_time_lora_scale=self.last_time_lora_scale, origin_weight=self.origin_weight)

        all_videos = sample_text2video(self.model, input_text, n_samples=1, batch_size=1,
                        sample_type='ddim', sampler=self.ddim_sampler,
                        ddim_steps=steps, eta=eta, 
                        cfg_scale=cfg_scale,
                        )
        prompt = input_text
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
        self.last_time_lora=self.lora_path_list[model_index]
        self.last_time_lora_scale = lora_scale
        video_path_list = save_results(all_videos, self.result_dir, save_name=prompt_str, save_fps=self.save_fps)
        return video_path_list[0]
    
    def download_model(self):
        REPO_ID = 'VideoCrafter/t2v-version-1-1'
        filename_list = ['models/base_t2v/model.ckpt',
                        'models/videolora/lora_001_Loving_Vincent_style.ckpt',
                        'models/videolora/lora_002_frozenmovie_style.ckpt',
                        'models/videolora/lora_003_MakotoShinkaiYourName_style.ckpt',
                        'models/videolora/lora_004_coco_style.ckpt']
        for filename in filename_list:
            if not os.path.exists(filename):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./', local_dir_use_symlinks=False)



def videocrafter_demo(result_dir='./tmp/'):
    text2video = Text2Video(result_dir)
    with gr.Blocks(analytics_enabled=False) as videocrafter_iface:
        gr.Markdown("<div align='center'> <h2> VideoCrafter: A Toolkit for Text-to-Video Generation and Editing </span> </h2> \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/VideoCrafter/VideoCrafter'> Github </div>")
        with gr.Row().style(equal_height=False):
            with gr.Tab(label="VideoCrafter"):
                input_text = gr.Text(label='Prompts')
                model_choices=['origin','vangogh','frozen','yourname', 'coco']

                with gr.Row():
                    model_index = gr.Dropdown(label='Models', elem_id=f"model", choices=model_choices, value=model_choices[0], type="index",interactive=True)
                    
                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=200, step=1, elem_id=f"steps", label="Sampling steps", value=50)
                    eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="eta")

                with gr.Row():
                    lora_scale = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label='Lora Scale', value=1.0, elem_id="lora_scale")
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=15.0, elem_id="cfg_scale")

                send_btn = gr.Button("Send")

            with gr.Column():
                output_video_1 = gr.PlayableVideo()

        with gr.Row():
            examples = [
                [
                    'an elephant is walking under the sea, 4K, high definition',
                    50,
                    'origin',
                    1,
                    15,
                    1,
                ],
                [
                    'an astronaut riding a horse in outer space',
                    25,
                    'origin',
                    1,
                    15,
                    1,
                ],
                [
                    'a monkey is playing a piano',
                    25,
                    'vangogh',
                    1,
                    15,
                    1,
                ],
                [
                    'A fire is burning on a candle',
                    25,
                    'frozen',
                    1,
                    15,
                    1,
                ],
                [
                    'a horse is drinking in the river',
                    25,
                    'yourname',
                    1,
                    15,
                    1,
                ],
                [
                    'Robot dancing in times square',
                    25,
                    'coco',
                    1,
                    15,
                    1,
                ],                    

            ]
            gr.Examples(examples=examples,
                        inputs=[
                        input_text,
                        steps,
                        model_index,
                        eta,
                        cfg_scale,
                        lora_scale],
                        outputs=[output_video_1],
                        fn=text2video.get_prompt,
                        cache_examples=False)

            send_btn.click(
                fn=text2video.get_prompt, 
                inputs=[
                    input_text,
                    steps,
                    model_index,
                    eta,
                    cfg_scale,
                    lora_scale,
                ],
                outputs=[output_video_1],
            )
    return videocrafter_iface

if __name__ == "__main__":
    result_dir = os.path.join('./', 'results')
    videocrafter_iface = videocrafter_demo(result_dir)
    videocrafter_iface.launch()