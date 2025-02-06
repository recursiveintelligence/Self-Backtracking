# Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models


A novel self-backtracking method for improving language model reasoning.

![Self-Backtracking Method](images/self-backtracking-method.jpg)

## Overview
This repository implements the Self-BackTracking method, that equips LLMs with the ability to backtrack during both training and inference. This mechanism not only enhances reasoning ability but also efficiency by transforming slow-thinking processes into fast-thinking through self-improvement.

## Dataset
The dataset used in this project is the Countdown dataset, which is pre-constructed and available on Hugging Face:
- Dataset: [yangxw/countdown-backtracking](https://huggingface.co/datasets/yangxw/countdown-backtracking)

## Getting Started

### Training
To train the model:

``` bash
python train.py --config ../configs/sft.conf
```