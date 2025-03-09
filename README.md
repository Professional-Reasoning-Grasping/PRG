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
- Vision-Language Model: [LLaVA-7B-v1-1](https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1)(F)
- Visual Backbone: [ViT-H SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (Fenc)
- Projection Layer: MLP (γ)
- Grasp Pose Decoder: Fdec
Training Process: Freeze Fenc, train Fdec, fine-tune F using AdaptiveMixture module

### Training Data
- PRG-4K (for professional reasoning grasping and long-text processing capabilities)
- [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json) (to retain original LLaVA visual reasoning capabilities)



## Dataset
The PRG-4K dataset will be made publicly available after the publication of the paper. lf you are interested in accessingthe data before then, please contact the author via email at maocheng@stumail.neu.edu.cn

