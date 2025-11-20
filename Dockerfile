FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Core tools + SSH + Python 3.11
RUN apt-get update && apt-get install -y \
    openssh-server supervisor nginx \
    python3.11 python3.11-dev python3.11-distutils python3.11-venv python3-pip \
    git curl wget nano vim ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN python -m pip install --upgrade pip

# JAX (GPU)
RUN pip install \
    jax==0.4.29 \
    jaxlib==0.4.29+cuda12.cudnn91 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# SSH configuration
RUN mkdir /var/run/sshd && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

# Workspace
WORKDIR /workspace

# RunPod entrypoint
ADD https://raw.githubusercontent.com/runpod/runpod-worker-utils/main/runpod-entrypoint.sh /runpod-entrypoint.sh
RUN chmod +x /runpod-entrypoint.sh

ENTRYPOINT ["/runpod-entrypoint.sh"]
