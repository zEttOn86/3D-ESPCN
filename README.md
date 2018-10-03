# 3D Efficient Subpixel-shifted Convolutional Network (3D-ESPCN)

This chainer implementation is based on "Tanno R. et al. (2017) Bayesian Image Quality Transfer with CNNs: Exploring Uncertainty in dMRI Super-Resolution. In: Descoteaux M., Maier-Hein L., Franz A., Jannin P., Collins D., Duchesne S. (eds) Medical Image Computing and Computer Assisted Intervention − MICCAI 2017. Lecture Notes in Computer Science, vol 10433. Springer, Cham".

Note that this is not official implementation.

The difference between original paper and this as follow:

- Dataset

  - I used [Balloon Analog Risk-taking Task dataset](https://openneuro.org/datasets/ds000001/versions/00006)

- I only implement baseline model (3D-ESPCN).

  - Network architecture

    ![Figure1](assets/img/figure1.png)

- Definition of pixel shuffler

  I think this is correct definition.

  <img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;S(F)_{i,j,k,c}&space;=&space;F_{[\frac{i}{r}],[\frac{j}{r}][\frac{k}{r}],(r^3-1)c&plus;mod(i,r)&plus;r&space;\cdot&space;mod(j,r)&plus;r^2&space;\cdot&space;mod(k,r)}&space;\end{align*}" />

  F: input feature map

  c: number of output image channel

  S: Pixel shuffler

  i, j, k, c: coordinate in output image

  r: upsampling rate

## Requirements

- chainer

- cupy

- SimpleITK

- pyyaml

## How to use

1. Download dataset [here](https://openneuro.org/datasets/ds000001/versions/00006).

    Please put all dataset to `data/raw` after you unzipped it.

2. Make mhd data and LR image

    ```
    # Make mhd data in data/interim
    python util\miscs\clean_data.py

    # Make LR and HR images in data/processed
    python util\miscs\make_lr_img.py
    ```

    - LR image sample (x1/4)

    ![Figure2](assets/img/LR_image_sample.png)

    - HR image sample

    ![Figure3](assets/img/HR_image_sample.png)


3. Train model

    ```
    python training.py -g 0
    ```

    - Training result

      ![Figure2](assets/img/gen_loss.png)

4. Infer HR image

    ```

    ```

5. Evaluations using PSNR and SSIM
