# Base image
FROM nvidia/cudagl:10.1-devel-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    iproute2 \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.9 cmake=3.14.0

# Add user
RUN useradd -u 1004 --create-home carlos
WORKDIR /home/carlos
RUN mkdir code/

# Setup habitat-sim
RUN git clone --depth 1 --branch v0.2.2 https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate habitat; cd habitat-sim; pip install -r requirements.txt; python setup.py install --headless"

# Install torch
RUN /bin/bash -c ". activate habitat; pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url  \
    https://download.pytorch.org/whl/cu113"

# This is to avoid breaking the offnav instalation
RUN /bin/bash -c ". activate habitat; pip install gym==0.22.0 setuptools==50.0.0"

# Install specific habitat-lab
RUN git clone --depth 1 --branch v0.2.2 https://github.com/facebookresearch/habitat-lab.git
RUN /bin/bash -c ". activate habitat; cd habitat-lab; python setup.py develop --all"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"
ENV HABITAT_SIM_LOG=quiet

# Change user
USER carlos

COPY entrypoint.sh /home/carlos
ENTRYPOINT [ "/home/carlos/entrypoint.sh" ]