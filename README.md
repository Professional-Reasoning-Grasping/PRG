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
![Model introduction and real machine experiments](https://github.com/Professional-Reasoning-Grasping/PRG/blob/main/image/video.mp4))
