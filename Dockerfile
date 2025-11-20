FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-distutils python3.11-venv python3-pip \
    git curl && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN python -m pip install --upgrade pip

RUN pip install \
    jax==0.4.29 \
    jaxlib==0.4.29+cuda12.cudnn91 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /workspace

# IMPORTANT: Keep container alive so RunPod SSH works
CMD ["sleep", "infinity"]
