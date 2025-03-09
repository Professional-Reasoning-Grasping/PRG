# Professional Reasoning Grasping via Multimodal Retrieval-Augmented Vision-Language Model

<img src="image/fig1.png" width="100">

We have uploaded the model code and related videos for reference during the review process.

More updates to follow soon! 

# Video
https://github.com/user-attachments/assets/fe22cf8a-b482-4024-b2a8-fc0a8ef4de29


## Installation
```pip install -r requirements.txt```


## Training
### Trainable Parameters
We adopt [LLaVA-7B-v1-1](https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1) as
the Vision-Language Model F , and use [ViT-H SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) vision
encoder as the visual backbone network Fenc. The projection
layer γ consists of an MLP with channel dimensions of
[256, 4096, 4096]. During training, we completely freeze
the visual backbone network Fenc and train the decoder
Fdec, while efficiently fine-tuning the model F using the
AdaptiveMixture module (see Section IV-B). Additionally,
the word embeddings and projection layer γ of the LLM are
also trainable.
### Training Data
Our model uses the PRG-4K training
set (see Section III-B) to enable its reasoning-based grasping
capabilities in professional domains and adaptability to
retrieving long texts. It also incorporates the [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json) (the original data used for LLaVA training)
to maintain the original VLMs’ visual reasoning capabilities.

Training scripts coming soon. We are in the process of refining our scripts to ensure they are efficient and user-friendly. Stay tuned for updates, and thank you for your interest and patience.


## Dataset
The PRG-4K dataset will be made publicly available after the publication of the paper. lf you are interested in accessingthe data before then, please contact the author via email at maocheng@stumail.neu.edu.cn

