#ARG PYTORCH="1.6.0"
#ARG CUDA="10.1"
#ARG PYTORCH="1.6.0"
#ARG CUDA="10.1"
#ARG CUDNN="7"

#FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

FROM nvcr.io/nvidia/pytorch:20.12-py3

#ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
#ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
#ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
#RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

RUN git clone https://gitee.com/cs20/mmcv.git /mmcv
WORKDIR /mmcv
ENV MMCV_WITH_OPS=1
RUN pip install -e .

# Install MMDetection
RUN conda clean --all
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
RUN git clone https://gitee.com/cs20/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA=1
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

WORKDIR /mmcv
ENV MMCV_WITH_OPS=1
RUN pip install -e .

WORKDIR /

