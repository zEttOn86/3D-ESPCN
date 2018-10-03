# FROM は Docker に対して基となるイメージを伝えます
# BEAR用
FROM nvidia/cuda:8.0-cudnn6-devel

# RUN 命令はイメージの中で実行するコマンドを指示．RUNはなるべく，ワンラインで書くこと
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    sudo git wget curl unzip tree graphviz && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir \
    cupy-cuda80 \
    chainer \
    matplotlib \
    scipy \
    scikit-learn\
    pandas \
    SimpleITK \
    tqdm \
    pyyaml


# userの追加
RUN groupadd -g 1002 developer && \
    useradd -g developer -u 1002 -G sudo -m -s /bin/bash penguin && \
    echo "penguin:piyopiyo" | chpasswd && \
    echo "penguin ALL=(ALL) ALL" >> /etc/sudoers

USER penguin
