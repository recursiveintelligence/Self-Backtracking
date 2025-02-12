# Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models

[![Arxiv](https://img.shields.io/badge/arXiv-2502.04404-b31b1b.svg)](https://www.arxiv.org/abs/2502.04404)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-2502.04404-yellow.svg)](https://huggingface.co/papers/2502.04404)
[![GitHub last commit](https://img.shields.io/github/last-commit/LAMDASZ-ML/Self-BackTracking.svg)](https://github.com/LAMDASZ-ML/Self-BackTracking)
[![Home Page](https://img.shields.io/badge/Home-Page-blue.svg)](https://github.com/LAMDASZ-ML/Self-BackTracking)
[![GitHub stars](https://img.shields.io/github/stars/LAMDASZ-ML/Self-BackTracking.svg?style=social)](https://github.com/LAMDASZ-ML/Self-BackTracking/stargazers)

A novel self-backtracking method for improving language model reasoning.

![Self-Backtracking Method](images/self-backtracking-method.jpg)

## Overview
This repository implements the Self-BackTracking method, that equips LLMs with the ability to backtrack during both training and inference. This mechanism not only enhances reasoning ability but also efficiency by transforming slow-thinking processes into fast-thinking through self-improvement.

## Dataset and Model
The project utilizes the Countdown dataset, which is pre-constructed and accessible on Hugging Face. Additionally, we have open-sourced our trained model based on Llama-3.2-1B.

[![Dataset](https://img.shields.io/badge/Dataset-Countdown-blue.svg)](https://huggingface.co/datasets/yangxw/countdown-backtracking)
[![Model](https://img.shields.io/badge/Model-countdown--backtrack-blue.svg)](https://huggingface.co/yangxw/Llama-3.2-1B-countdown-backtrack)
## Getting Started

### Training
To train the model:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config ../configs/sft.conf
```
You can change the parameters in the `configs/sft.conf` file.

If you want to use multiple GPUs:
```bash
accelerate launch \
    --config_file ../configs/accelerate.yaml \
    train.py \
    --config ../configs/sft.conf
```

### Inference
To inference the model using our self-backtracking method, you can run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_search.py \
    --num 5000 \
    --ckpt [your_model_ckpt] \
    --data [val/val_new] \
    --decoder self_backtrack \
    --b 1 \
    --n 32
```
--ckpt defaults to `yangxw/Llama-3.2-1B-countdown-backtrack`. You can use our trained model available on Hugging Face.

### Self-Improvement
To further improve the model, you can run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python train_self_improvement.py \
    --num 5000 \
    --past_model [your_model_ckpt] \
    --data [val/val_new]
```

## Results
![Results](images/tab.png)
## Citation
If you use this work, please cite it as follows:

```
@article{selfbacktracking2023,
  title={Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models},
  author={Xiao-Wen Yang and Xuan-Yi Zhu and Wen-Da Wei and Ding-Chu Zhang and Jie-Jing Shao and Zhi Zhou and Lan-Zhe Guo and Yu-Feng Li},
  journal={arXiv preprint arXiv:2502.04404},
  year={2025}
}
```