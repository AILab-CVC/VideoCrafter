import os
import sys
import gradio as gr
from scripts.gradio.t2v_test import Text2Video
from scripts.gradio.i2v_test import Image2Video
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))

t2v_examples = [
    ['an elephant is walking under the sea, 4K, high definition',50, 12,1, 16],
    ['an astronaut riding a horse in outer space',25,12,1,16],
    ['a monkey is playing a piano',25,12,1,16],
    ['A fire is burning on a candle',25,12,1,16],
    ['a horse is drinking in the river',25,12,1,16],
    ['Robot dancing in times square',25,12,1,16],                    
]

i2v_examples = [
    ['prompts/i2v_prompts/horse.png', 'horses are walking on the grassland', 50, 12, 1, 16]
]

def videocrafter_demo(result_dir='./tmp/'):
    text2video = Text2Video(result_dir)
    image2video = Image2Video(result_dir)
    with gr.Blocks(analytics_enabled=False) as videocrafter_iface:
        gr.Markdown("<div align='center'> <h2> VideoCrafter1: Open Diffusion Models for High-Quality Video Generation </span> </h2> \
                     <a style='font-size:18px;color: #000000' href='https://github.com/AILab-CVC/VideoCrafter'> Github </div>")
        
        #######t2v#######
        with gr.Tab(label="Text2Video"):
            with gr.Column():
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        input_text = gr.Text(label='Prompts')
                        with gr.Row():
                            steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id=f"steps", label="Sampling steps", value=50)
                            eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="eta")
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=12.0, elem_id="cfg_scale")
                            fps = gr.Slider(minimum=4, maximum=32, step=1, label='fps', value=16, elem_id="fps")
                        send_btn = gr.Button("Send")
                    with gr.Tab(label='result'):
                        with gr.Row():
                            output_video_1 =  gr.Video().style(width=512)
                gr.Examples(examples=t2v_examples,
                            inputs=[input_text,steps,cfg_scale,eta],
                            outputs=[output_video_1],
                            fn=text2video.get_prompt,
                            cache_examples=False)
                        #cache_examples=os.getenv('SYSTEM') == 'spaces')
            send_btn.click(
                fn=text2video.get_prompt, 
                inputs=[input_text,steps,cfg_scale,eta,fps],
                outputs=[output_video_1],
            )
        #######image2video######
        with gr.Tab(label='Image2Video'):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            i2v_input_image = gr.Image(label="Input Image").style(width=256)
                        with gr.Row():
                            i2v_input_text = gr.Text(label='Prompts')
                        with gr.Row():
                            i2v_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="i2v_eta")
                            i2v_cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=12.0, elem_id="i2v_cfg_scale")
                        with gr.Row():
                            i2v_steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                            i2v_fps = gr.Slider(minimum=4, maximum=32, step=1, elem_id="i2v_fps", label="Generative fps", value=16)
                        i2v_end_btn = gr.Button("Send")
                    with gr.Tab(label='Result'):
                        with gr.Row():
                            i2v_output_video = gr.Video(label="Generated Video").style(width=512)

                gr.Examples(examples=i2v_examples,
                            inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_fps],
                            outputs=[i2v_output_video],
                            fn = image2video.get_image,
                            cache_examples=os.getenv('SYSTEM') == 'spaces',
                )
            i2v_end_btn.click(inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_fps],
                            outputs=[i2v_output_video],
                            fn = image2video.get_image
            )

    return videocrafter_iface

if __name__ == "__main__":
    result_dir = os.path.join('./', 'results')
    videocrafter_iface = videocrafter_demo(result_dir)
    videocrafter_iface.queue(concurrency_count=1, max_size=10)
    videocrafter_iface.launch()
    # videocrafter_iface.launch(server_name='0.0.0.0', server_port=80)