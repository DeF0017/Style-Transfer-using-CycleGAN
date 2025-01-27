# Style-Transfer-using-CycleGAN

This repository contains code for performing image style transfer using CycleGAN. CycleGAN is a model that enables image-to-image translation without requiring paired examples. It can be used to transfer styles between two different domains, such as converting photos to paintings or vice versa.

Huggingface Space: [link](https://huggingface.co/spaces/DeF0017/Style-Transfer-using-CycleGAN)

![Photo](https://github.com/DeF0017/Style-Transfer-using-CycleGAN/blob/main/ist_cyclegan2.png)

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Results](#results)
- [References](#references)

## Overview

CycleGAN learns to translate images from one domain (e.g., photos) to another domain (e.g., paintings) without the need for paired training examples. This is achieved using two generator networks and two discriminator networks:

- Generator \(G\): Translates images from domain X to domain Y.
- Generator \(F\): Translates images from domain Y to domain X.
- Discriminator \(D_X\): Distinguishes real images in domain X from translated images.
- Discriminator \(D_Y\): Distinguishes real images in domain Y from translated images.

The model is trained with the following losses:
- Adversarial Loss: Ensures the generators produce realistic images.
- Cycle Consistency Loss: Ensures that translating an image to the other domain and back yields the original image.
- Identity Loss: Ensures that images in the target domain are not modified when they are already in the target domain.

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision 0.9+
- numpy
- matplotlib
- tqdm
- Pillow

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/cyclegan-style-transfer.git
cd cyclegan-style-transfer
pip install -r requirements.txt
```
## Dataset

For this project, the [Van Gogh to Photo dataset](https://www.kaggle.com/datasets/def0017/vangogh2photo) was used. This dataset contains images of Van Gogh paintings and corresponding photos, which are used to train the CycleGAN model for style transfer between these two domains.

### Download and Setup

To download and set up the dataset, follow these steps:

1. **Download the Dataset**: Download the Van Gogh to Photo dataset from [this link](https://www.kaggle.com/datasets/def0017/vangogh2photo).

2. **Extract the Dataset**: Extract the dataset into the `./datasets` directory. After extracting, your directory structure should look like this:
    ```
    ./datasets/vangogh2photo/vangogh2photo/
        trainA/
        trainB/
        testA/
        testB/
    ```

    - `trainA/`: Contains training images from domain A (e.g., photos).
    - `trainB/`: Contains training images from domain B (e.g., Van Gogh paintings).
    - `testA/`: Contains test images from domain A (e.g., photos).
    - `testB/`: Contains test images from domain B (e.g., Van Gogh paintings).

## Results

![Photo](https://github.com/DeF0017/Style-Transfer-using-CycleGAN/blob/main/ist_cyclegan.png)

### Interpreting the Results

- **Input (Photo) to Translated (Van Gogh)**: This images shows how well the model can transform a normal photo into a Van Gogh-style painting. [Click Here](https://github.com/DeF0017/Style-Transfer-using-CycleGAN/tree/main/Images_A)
- **Input (Van Gogh) to Translated (Photo)**: This images shows the reverse transformation, converting a Van Gogh painting into a photo-like image. [Click Here](https://github.com/DeF0017/Style-Transfer-using-CycleGAN/tree/main/Images_B)

These transformations demonstrate the model's ability to learn and apply complex artistic styles across different domains.

Huggingface Space: [link](https://huggingface.co/spaces/DeF0017/Style-Transfer-using-CycleGAN)

## References

If you use this code or find it helpful, please consider citing the following papers and resources that were instrumental in the development of the CycleGAN style transfer project:

### Primary Paper

- **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**
  - Authors: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
  - Paper: [arXiv preprint arXiv:1703.10593](https://arxiv.org/abs/1703.10593)

These references and resources provide the foundational knowledge and tools used in developing and understanding CycleGAN for style transfer. If you build upon this project, please acknowledge the original authors and resources accordingly.
