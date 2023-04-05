
# VideoCrafter：A Toolkit for Text-to-Video Generation and Editing 


## 🔆 Introduction
<img src=assets/intro.gif>  

🤗🤗🤗 VideoCrafter is an open-source video generation and editing toolbox for crafting video content.   
It currently includes the following THREE types of models:

### 1. Base T2V: Generic Text-to-video Generation
We provide a base text-to-video (T2V) generation model based on the latent video diffusion models ([LVDM](https://yingqinghe.github.io/LVDM/)). 
It can synthesize realistic videos based on the input text descriptions.

<table class="center">
  <td style="text-align:center;" width="170">"Campfire at night in a snowy forest with starry sky in the background."</td>
  <td style="text-align:center;" width="170">"Cars running on the highway at night."</td>
  <td style="text-align:center;" width="170">"close up of a clown fish swimming. 4K"</td>
  <td style="text-align:center;" width="170">"astronaut riding a horse"</td>
  <tr>
  <td><img src=assets/summary/t2v/005.gif width="170"></td>
  <td><img src=assets/summary/t2v/006.gif width="170"></td>
  <td><img src=assets/summary/t2v/008.gif width="170"></td>
  <td><img src=assets/summary/t2v/004.gif width="170"></td>
</table >

<!-- <br>   -->

### 2. VideoLoRA: Personalized Text-to-Video Generation with LoRA

Based on the pretrained LVDM, we can create our **own** video generation models by finetuning it on a set of video clips or images describing a certain concept.

We adopt [LoRA](https://arxiv.org/abs/2106.09685) to implement the finetuning as it is eary to train and require less computational resources.

Below are generation results from our **four VideoLoRA models** that are trained on four different styles of video clips.

By providing a sentence describing the video content along with a LoRA trigger word (specified during LoRA training), it can generate videos with the desired style(or subject/concept).


Results of inputting `A monkey is playing a piano, ${trigger_word}` to the four VideoLoRA models:   
<table class="center">
  <td><img src=assets/summary/1/001_loving_vincent.gif width="170"></td>
  <td><img src=assets/summary/1/002_frozen.gif width="170"></td>
  <td><img src=assets/summary/1/003_your_name.gif width="170"></td>
  <td><img src=assets/summary/1/004_coco.gif width="170"></td>
  </tr>
  <td style="text-align:center;" width="170">"Loving Vincent style"</td>
  <td style="text-align:center;" width="170">"frozenmovie style"</td>
  <td style="text-align:center;" width="170">"MakotoShinkaiYourName style"</td>
  <td style="text-align:center;" width="170">"coco style"</td>
  <tr>
</table >
The trigger word for each VideoLoRA is annotated below the generation result.  

<br>  

### 3. VideoControl: Video Generation with More Condition Controls
To enhance the controllable abilities of the T2V model, we developed conditional adapter inspired by [T2I-adapter](https://github.com/TencentARC/T2I-Adapter).
By pluging a lightweight adapter module to the T2V model, we can obtained generation results with more detailed control signals such as depth.

input text: `Ironman is fighting against the enemy, big fire in the background, photorealistic, 4k`
<table class="center">
  <td><img src=assets/adapter/6_GIF/input_5_randk1.gif width="170"></td>
  <td><img src=assets/adapter/6_GIF/depth_5_randk1.gif width="170"></td>
  <td><img src=assets/adapter/6_GIF/0006.gif width="170"></td>
  <td><img src=assets/adapter/6_GIF/0008.gif width="170"></td>
  <td><img src=assets/adapter/6_GIF/0004.gif width="170"></td>
  </tr>
</table >


🤗🤗🤗 We will keep updating this repo and add more features and models. Please stay tuned!

</table >

<br>  

---

## 📝 Changelog
- __[2023.04.05]__: release pretrained Text-to-Video models, VideoLora models, and inference code.
<br>

<!--  -->
## ⏳ TODO
- [ ] Huggingface demo
- [ ] Release training code for VideoLoRA
- [ ] Technical report
- [ ] More customized models

<br>  


---
## ⚙️ Setup


<!-- <details><summary>CLICK ME for installing environment via Anaconda </summary> -->

### Installing environment via Anaconda
```bash
conda create -n lvdm python=3.8.5
conda activate lvdm
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.8.3 omegaconf==2.1.1 einops==0.3.0 transformers==4.25.1
pip install opencv-python==4.1.2.30 imageio==2.9.0 imageio-ffmpeg==0.4.2
pip install av moviepy
pip install -e .
```

</details>


<br>  

## 💫 Inference 
### 1. Text-to-Video

1) Download pretrained T2V models via this [link](https://drive.google.com/file/d/13ZZTXyAKM3x0tObRQOQWdtnrI2ARWYf_/view?usp=share_link), and put the `model.ckpt` in `models/base_t2v/model.ckpt`.
2) Input the following commands in terminal, it will start running in the GPU 0.
```bash
  PROMPT="astronaut riding a horse" 
  OUTDIR="results/"

  BASE_PATH="models/base_t2v/model.ckpt"
  CONFIG_PATH="models/base_t2v/model_config.yaml"

  python scripts/sample_text2video.py \
      --ckpt_path $BASE_PATH \
      --config_path $CONFIG_PATH \
      --prompt "$PROMPT" \
      --save_dir $OUTDIR \
      --n_samples 1 \
      --batch_size 1 \
      --seed 1000 \
      --show_denoising_progress
```


<details><summary>CLICK ME for more options </summary>

- `gpu_id`: specify the gpu index you want to use.  
- `ddp`: better to enable it if you have multiple GPUs.

</details>


<!-- <br> -->


### 2. VideoLoRA
1) Same with 1-1: Download pretrained T2V models via this [link](https://drive.google.com/file/d/13ZZTXyAKM3x0tObRQOQWdtnrI2ARWYf_/view?usp=share_link), and put the `model.ckpt` in `models/base_t2v/model.ckpt`.
   
2) Download pretrained VideoLoRA models via this [link](https://drive.google.com/drive/folders/14tK8K_-3aLIrDIrr5CeUxzhGHn5gYBUZ?usp=share_link) (can select one model), and put it in `models/videolora/${model_name}.ckpt`.

3) Input the following commands in terminal, it will start running in the GPU 0.

```bash
  PROMPT="astronaut riding a horse"
  OUTDIR="results/videolora"

  BASE_PATH="models/base_t2v/model.ckpt"
  CONFIG_PATH="models/base_t2v/model_config.yaml"

  LORA_PATH="models/videolora/lora_001_Loving_Vincent_style.ckpt"
  TAG=", Loving Vincent style"

  python scripts/sample_text2video.py \
      --ckpt_path $BASE_PATH \
      --config_path $CONFIG_PATH \
      --prompt "$PROMPT" \
      --save_dir $OUTDIR \
      --n_samples 1 \
      --batch_size 1 \
      --seed 1000 \
      --show_denoising_progress \
      --inject_lora \
      --lora_path $LORA_PATH \
      --lora_trigger_word "$TAG" \
      --lora_scale 1.0
```


4) If your find the lora effect is either too large or too small, you can adjust the `lora_scale` argument to control the strength.
   <details><summary>CLICK ME for the visualization of different lora scales </summary>
   
    The effect of LoRA weights can be controlled by the `lora_scale`. `local_scale=0` indicates using the original base model, while `local_scale=1` indicates using the full lora weights. It can also be slightly larger than 1 to emphasize more effect from lora.

    <table class="center">
      <td style="text-align:center;" width="170">scale=0.0</td>
      <td style="text-align:center;" width="170">scale=0.25</td>
      <td style="text-align:center;" width="170">scale=0.5</td>
      <tr>
      <td><img src=assets/diffscale/astronaut_riding_a_horse,_Loving_Vincent_style_000_scale0.gif width="170"></td>
      <td><img src=assets/diffscale/astronaut_riding_a_horse,_Loving_Vincent_style_000_scale0.25.gif width="170"></td>
      <td><img src=assets/diffscale/astronaut_riding_a_horse,_Loving_Vincent_style_000_scale0.5.gif width="170"></td>
      </tr>
      <td style="text-align:center;" width="170">scale=0.75</td>
      <td style="text-align:center;" width="170">scale=1.0</td>
      <td style="text-align:center;" width="170">scale=1.5</td>
      <tr>
      <td><img src=assets/diffscale/astronaut_riding_a_horse,_Loving_Vincent_style_000_scale0.75.gif width="170"></td>
      <td><img src=assets/diffscale/astronaut_riding_a_horse,_Loving_Vincent_style_000_scale1.0.gif width="170"></td>
      <td><img src=assets/diffscale/astronaut_riding_a_horse,_Loving_Vincent_style_000_scale1.5.gif width="170"></td>
    </table >

</details>

### 3. VideoControl
⏳⏳⏳ Comming soon. Please stay tuned!🤗

<br>

---
## 🥳 Gallery
### VideoLoRA Models
#### Loving Vincent Style
<table class="center">
  <!-- <td style="text-align:center;" width="50">Input Text</td> -->
  <td style="text-align:center;" width="170">"A blue unicorn flying over a mystical land"</td>
  <td style="text-align:center;" width="170">"A teddy bear washing the dishes"</td>
  <td style="text-align:center;" width="170">"Flying through an intense battle between pirate ships in a stormy ocean"</td>
  <td style="text-align:center;" width="170">"a rabbit driving a bicycle, in Tokyo at night"</td>
  <tr>
  <td><img src=assets/lora/1_vangogh/013.gif width="170"></td>
  <td><img src=assets/lora/1_vangogh/002.gif width="170"></td>
  <td><img src=assets/lora/1_vangogh/001.gif width="170"></td>
  <td><img src=assets/lora/1_vangogh/011.gif width="170"></td>
</tr>
</table >

#### Frozen
<table class="center">
  <!-- <td style="text-align:center;" width="50">Input Text</td> -->
  <td style="text-align:center;" width="170">"A fire is burning on a candle."</td>
  <td style="text-align:center;" width="170">"A giant spaceship is landing on mars in the sunset. High Definition."</td>
  <td style="text-align:center;" width="170">"A bear dancing and jumping to upbeat music, moving his whole body."</td>
  <td style="text-align:center;" width="170">"Face of happy macho mature man smiling."</td>
  <tr>
  <td><img src=assets/lora/2_frozen/001.gif width="170"></td>
  <td><img src=assets/lora/2_frozen/012.gif width="170"></td>
  <td><img src=assets/lora/2_frozen/011.gif width="170"></td>
  <td><img src=assets/lora/2_frozen/004.gif width="170"></td>
</tr>
</table >

#### Your Name
<table class="center">
  <!-- <td style="text-align:center;" width="50">Input Text</td> -->
  <td style="text-align:center;" width="170">"A man playing a saxophone with musical notes flying out."</td>
  <td style="text-align:center;" width="170">"Flying through an intense battle between pirate ships in a stormy ocean"</td>
  <td style="text-align:center;" width="170">"Horse drinking water."</td>
  <td style="text-align:center;" width="170">"Woman in sunset."</td>
  <tr>
  <td><img src=assets/lora/3_your_name/012.gif width="170"></td>
  <td><img src=assets/lora/3_your_name/011.gif width="170"></td>
  <td><img src=assets/lora/3_your_name/007.gif width="170"></td>
  <td><img src=assets/lora/3_your_name/013.gif width="170"></td>
</tr>
</table >

#### CoCo
<table class="center">
  <td style="text-align:center;" width="170">"Humans building a highway on mars, highly detailed"</td>
  <td style="text-align:center;" width="170">"A blue unicorn flying over a mystical land"</td>
  <td style="text-align:center;" width="170">"Robot dancing in times square"</td>
  <td style="text-align:center;" width="170">"A 3D model of an elephant origami. Studio lighting."</td>
  <tr>
  <td><img src=assets/lora/4_coco/008.gif width="170"></td>
  <td><img src=assets/lora/4_coco/005.gif width="170"></td>
  <td><img src=assets/lora/4_coco/009.gif width="170"></td>
  <td><img src=assets/lora/4_coco/001.gif width="170"></td>
</tr>
</table >



### VideoControl

<table class="center">
  <td colspan="5" >"A camel walking on the snow field, Miyazaki Hayao anime style"</td>
  </tr>
  <td><img src=assets/adapter/5_GIF/input_4_randk0.gif width="170"></td>
  <td><img src=assets/adapter/5_GIF/depth_4_randk0.gif width="170"></td>
  <td><img src=assets/adapter/5_GIF/0000.gif width="170"></td>
  <td><img src=assets/adapter/5_GIF/0008.gif width="170"></td>
  <td><img src=assets/adapter/5_GIF/0006.gif width="170"></td>
  <!-- <td><img src=assets/adapter/5_GIF/0001.gif width="170"></td> -->
  </tr>
  <td colspan="5" >"Ironman playing hockey on the field, photorealistic, 4k"</td>
  </tr>
  <td><img src=assets/adapter/2_GIF/input_2_randk1.gif width="170"></td>
  <td><img src=assets/adapter/2_GIF/depth_2_randk1.gif width="170"></td>
  <td><img src=assets/adapter/2_GIF/0003.gif width="170"></td>
  <td><img src=assets/adapter/2_GIF/0004.gif width="170"></td>
  <td><img src=assets/adapter/2_GIF/0008.gif width="170"></td>
  </tr>
  
  <td colspan="5" >"An ostrich walking in the desert, photorealistic, 4k"</td>
  </tr>
  <td><img src=assets/adapter/1_GIF/input_1_randk1.gif width="170"></td>
  <td><img src=assets/adapter/1_GIF/depth_1_randk1.gif width="170"></td>
  <td><img src=assets/adapter/1_GIF/0003.gif width="170"></td>
  <td><img src=assets/adapter/1_GIF/0002.gif width="170"></td>
  <td><img src=assets/adapter/1_GIF/0009.gif width="170"></td>
  </tr>
  <td colspan="5" >"A car turning around on a countryside road, snowing heavily, ink wash painting"</td>
  </tr>
  <td><img src=assets/adapter/7_GIF/input_5_randk0.gif width="170"></td>
  <td><img src=assets/adapter/7_GIF/depth_5_randk0.gif width="170"></td>
  <td><img src=assets/adapter/7_GIF/0003.gif width="170"></td>
  <td><img src=assets/adapter/7_GIF/0004.gif width="170"></td>
  <td><img src=assets/adapter/7_GIF/0009.gif width="170"></td>
  </tr>
  
</table >


---
## 📋 Techinical Report
⏳⏳⏳ Comming soon. We are still working on it.💪
<br>

<!-- ## 💗 Related Works -->
## 📭 Contact
If your have any comments or questions, feel free to contact [Yingqing He](yhebm@connect.ust.hk), [Haoxin Chen](jszxchx@126.com) or [Menghan Xia](menghanxyz@gmail.com).

## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****