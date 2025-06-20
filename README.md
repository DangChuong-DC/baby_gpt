# Baby GPT

## Overview
Baby GPT is a simple implementation of a Generative Pre-trained Transformer (GPT) model. It is designed to be easy to understand and modify, making it a great starting point for those interested in learning about transformer models and natural language processing.

This is inspired greatly by the work of Andrej Karpathy and his [minGPT](https://github.com/karpathy/nanoGPT)

## How to Use

### To train the model:
1. Install the required packages
2. Run the training script:
```bash
python train.py
```

### To generate text:
1. Install the required packages
2. Run the generation script:
```bash
python inference.py
```

## Model Architecture

#### General Architecture
![plot](./assets/model_architecture.png)

#### Detailed Architecture
![plot](./assets/Full_GPT_architecture.png)

## Data
The model is trained on a small dataset of text, which can be found in the `data` directory. The dataset is a collection of text which is poem of Xuan Dieu, a famous Vietnamese poet.
I copied the text from [this website](https://www.thivien.net/) by searching for Xuân Diệu, then saved it in a file called `data/xuandieutho.txt`. The dataset is small, but it is enough to demonstrate the capabilities of the model. I might add more data in the future if I have time.
