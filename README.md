### 3D Efficient Subpixel-shifted Convolutional Network (3D-ESPCN)

This chainer implementation is based on "Tanno R. et al. (2017) Bayesian Image Quality Transfer with CNNs: Exploring Uncertainty in dMRI Super-Resolution. In: Descoteaux M., Maier-Hein L., Franz A., Jannin P., Collins D., Duchesne S. (eds) Medical Image Computing and Computer Assisted Intervention − MICCAI 2017. Lecture Notes in Computer Science, vol 10433. Springer, Cham".

Note that this is not official implementation.

The difference between original paper and this as follow:

- Dataset

  - I used [Balloon Analog Risk-taking Task dataset](https://openneuro.org/datasets/ds000001/versions/00006)

- I only implement baseline model (3D-ESPCN).

  - Network architecture

#### Requirements

- chainer

- cupy

- SimpleITK

- pyyaml

#### How to use

1. Download dataset [here](https://openneuro.org/datasets/ds000001/versions/00006).

    Please put all dataset to `data/raw` after you unzipped it.

2. Make mhd data and LR image

    ```
    # Make mhd data in data/interim
    python util\miscs\clean_data.py

    # Make LR and HR images in data/processed
    python util\miscs\make_lr_img.py
    ```

3. Training model

    ```
    python training.py -g 0
    ```

4. Reconstruct HR image

    ```
    
    ```
