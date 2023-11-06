
## ___***VideoCrafter1: Open Diffusion Models for High-Quality Video Generation***___

<a href='https://ailab-cvc.github.io/videocrafter/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/abs/2310.19512'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
[![Discord](https://dcbadge.vercel.app/api/server/rrayYqZ4tf?style=flat)](https://discord.gg/rrayYqZ4tf)
<a href='https://huggingface.co/spaces/VideoCrafter/VideoCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
[![Replicate](https://replicate.com/cjwbw/videocrafter/badge)](https://replicate.com/cjwbw/videocrafter)
[![GitHub](https://img.shields.io/github/stars/VideoCrafter/VideoCrafter?style=social)](https://github.com/VideoCrafter/VideoCrafter)


### 🔥🔥 The VideoCrafter1 for high-quality video generation are now released!  Please Join us and create your own film on [Discord/Floor33](https://discord.gg/rrayYqZ4tf).

### Floor33 | Film
 [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/4MbTNYug1wo/0.jpg)](https://www.youtube.com/watch?v=4MbTNYug1wo)
 
## 🔆 Introduction


🤗🤗🤗 VideoCrafter is an open-source video generation and editing toolbox for crafting video content.   
It currently includes the Text2Video and Image2Video models:

### 1. Generic Text-to-video Generation
Click the GIF to access the high-resolution video.

<table class="center">
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/814f9cfe-5e4c-4d6c-be4c-c378cf4216c7"><img src=assets/t2v/agirl.gif width="320"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/f89af8d2-2ac4-4726-98cc-4ff322ed4cf3"><img src=assets/t2v/astronaut.gif width="320"></td>
  <tr>
  <td style="text-align:center;" width="320">"A girl is looking at the camera smiling. High Definition."</td>
  <td style="text-align:center;" width="320">"an astronaut running away from a dust storm on the surface of the  moon, the astronaut is running towards the camera, cinematic"</td>
  <tr>
</table >

<table class="center">
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/232ef312-be08-4d73-8fd7-f367952c9410"><img src=assets/t2v/spaceship.gif width="320"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/7aa3977c-dc71-45ce-bfe2-449368dc1c9f"><img src=assets/t2v/unicorn.gif width="320"></td>
  <tr>
  <td style="text-align:center;" width="320">"A giant spaceship is landing on mars in the sunset. High Definition."</td>
  <td style="text-align:center;" width="320">"A blue unicorn flying over a mystical land"</td>
  <tr>
</table >

### 2. Generic Image-to-video Generation

<table class="center">
  <td><img src=assets/i2v/input/blackswan.png width="170"></td>
  <td><img src=assets/i2v/input/horse.png width="170"></td>
  <td><img src=assets/i2v/input/chair.png width="170"></td>
  <td><img src=assets/i2v/input/sunset.png width="170"></td>
  <tr>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/1a57edd9-3fd2-4ce9-8313-89aca95b6ec7"><img src=assets/i2v/blackswan.gif width="170"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/d671419d-ae49-4889-807e-b841aef60e8a"><img src=assets/i2v/horse.gif width="170"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/39d730d9-7b47-4132-bdae-4d18f3e651ee"><img src=assets/i2v/chair.gif width="170"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/dc8dd0d5-a80d-4f31-94db-f9ea0b13172b"><img src=assets/i2v/sunset.gif width="170"></td>
  <tr>
  <td style="text-align:center;" width="170">"a black swan swims on the pond"</td>
  <td style="text-align:center;" width="170">"a girl is riding a horse fast on grassland"</td>
  <td style="text-align:center;" width="170">"a boy sits on a chair facing the sea"</td>
  <td style="text-align:center;" width="170">"two galleons moving in the wind at sunset"</td>

</table >


---

## 📝 Changelog
- __[2023.10.30]__: Release [VideoCrafter1](https://arxiv.org/abs/2310.19512) Technical Report!

- __[2023.10.19]__: Release the 320x512 Text2Video Model, and HuggingFace demo.

- __[2023.10.13]__: 🔥🔥 Release the VideoCrafter1, High Quality Video Generation!

- __[2023.08.14]__: Release a new version of VideoCrafter on [Discord/Floor33](https://discord.gg/uHaQuThT). Please join us to create your own film!

- __[2023.04.18]__: Release a VideoControl model with most of the watermarks removed!

- __[2023.04.05]__: Release pretrained Text-to-Video models, VideoLora models, and inference code.
<br>


## ⏳ Models

|Models|Resolution|Checkpoints|
|:---------|:---------|:--------|
|Text2Video|576x1024|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt)
|Text2Video|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt)
|Image2Video|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt)



## ⚙️ Setup

### 1. Install Environment via Anaconda (Recommended)
```bash
conda create -n videocrafter python=3.8.5
conda activate videocrafter
pip install -r requirements.txt
```


## 💫 Inference 
### 1. Text-to-Video

1) Download pretrained T2V models via [Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024-v1.0/blob/main/model.ckpt), and put the `model.ckpt` in `checkpoints/base_1024_v1/model.ckpt`.
2) Input the following commands in terminal.
```bash
  sh scripts/run_text2video.sh
```

### 2. Image-to-Video

1) Download pretrained I2V models via [Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512-v1.0/blob/main/model.ckpt), and put the `model.ckpt` in `checkpoints/i2v_512_v1/model.ckpt`.
2) Input the following commands in terminal.
```bash
  sh scripts/run_image2video.sh
```

### 3. Local Gradio demo

1. Download the pretrained T2V and I2V models and put them in the corresponding directory according to the previous guidelines.
2. Input the following commands in terminal.
```bash
  python gradio_app.py
```

---
## 📋 Techinical Report
😉 Tech report: [VideoCrafter1: Open Diffusion Models for High-Quality Video Generation](https://arxiv.org/abs/2310.19512)
<br>

## 😉 Citation
The technical report is currently unavailable as it is still in preparation. You can cite the paper of our image-to-video model and related base model.
```
@misc{chen2023videocrafter1,
      title={VideoCrafter1: Open Diffusion Models for High-Quality Video Generation}, 
      author={Haoxin Chen and Menghan Xia and Yingqing He and Yong Zhang and Xiaodong Cun and Shaoshu Yang and Jinbo Xing and Yaofang Liu and Qifeng Chen and Xintao Wang and Chao Weng and Ying Shan},
      year={2023},
      eprint={2310.19512},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{xing2023dynamicrafter,
      title={DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors}, 
      author={Jinbo Xing and Menghan Xia and Yong Zhang and Haoxin Chen and Xintao Wang and Tien-Tsin Wong and Ying Shan},
      year={2023},
      eprint={2310.12190},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{he2022lvdm,
      title={Latent Video Diffusion Models for High-Fidelity Long Video Generation}, 
      author={Yingqing He and Tianyu Yang and Yong Zhang and Ying Shan and Qifeng Chen},
      year={2022},
      eprint={2211.13221},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## 🤗 Acknowledgements
Our codebase builds on [Stable Diffusion](https://github.com/Stability-AI/stablediffusion). 
Thanks the authors for sharing their awesome codebases! 


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****
