# Professional Reasoning Grasping via Multimodal Retrieval-Augmented Vision-Language Model

## Abstract
  The development of Vision-Language Models (VLMs) offers a new direction for cross-modal robotic grasping, enabling robots to perform grasping tasks based on visual-language instructions. 
However, most existing methods focus on everyday or general domains, making it difficult to reason about implicit instructions that require professional knowledge in fields such as electric power or mechanical engineering. 
This limits their application in high-standard, high-risk environments. 
To address this challenge, we propose the Professional Reasoning Grasping (PRG) task and construct the corresponding PRG-10K dataset. 
The dataset covers four industries—electric power, chemical, mechanical, and home appliance—and includes regulatory documents, images, Q\&A pairs, and annotated grasp poses. 
We further propose a grasping strategy that integrates multimodal Retrieval-Augmented Generation (RAG) with VLMs and adopt the Embedding as Pose paradigm to simultaneously output text and numerical grasp poses. 
In addition, we design the AdaptiveMixture fine-tuning module to enhance the model's adaptability to long texts and multimodal inputs. 
Experimental results show that our method outperforms existing VLM-based grasping approaches in target recognition, grasp success rate, text output quality, and zero-shot capability, thereby improving the reliability of robots in the PRG task and their ability to address implicit instructions in professional domains. 

## model
![The model generates grasp configurations and text
responses based on visual and language instructions, as well
as retrieved professional domain documents.](image/fig.png)

## Installation
```pip install -r requirements.txt```

## Training
### Trainable Parameters
We adopt LLaVA-7B-v1-1 as
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
Our model uses the PRG-10K training
set (see Section III-B) to enable its reasoning-based grasping
capabilities in professional domains and adaptability to
retrieving long texts. It also incorporates the [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json) (the original data used for LLaVA training)
to maintain the original VLMs’ visual reasoning capabilities.

3. Reasoning Grasp dataset: ReasonSeg

Download them from the above links, and organize them as follows.

```
├── dataset
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
```


#### Pre-trained weights
#### LLaVA
To train LISA-7B or 13B, you need to follow the [instruction](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) to merge the LLaVA delta weights. Typically, we use the final weights `LLaVA-Lightning-7B-v1-1` and `LLaVA-13B-v1-1` merged from `liuhaotian/LLaVA-Lightning-7B-delta-v1-1` and `liuhaotian/LLaVA-13b-delta-v1-1`, respectively. For Llama2, we can directly use the LLaVA full weights `liuhaotian/llava-llama-2-13b-chat-lightning-preview`.

### Training
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LLaVA" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="professional_grasp" \
  --exp_name="PRG-7b"
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```
### Validation
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LISA_HF_Model_Directory" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --exp_name="PRG-7b" \
  --eval_only
```

## Inference 
To run inference with the model, you will need to use the `chat.py` script. This script allows you to interact with the model in a chat-like interface where you can input your queries and receive responses. Below are the instructions on how to execute the script:
```
python chat.py
```

## Dataset
The datasets generated and/or analyzed during the current study are available from the author on reasonablerequest. The data will be made publicly available after the publication of the paper. lf you are interested in accessingthe data before then, please contact the author via email at maocheng@stumail.neu.edu.cn
