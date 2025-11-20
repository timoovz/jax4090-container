FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# --- Basic noninteractive environment ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- Install Python 3.11 and basic tools ---
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils python3.11-venv python3-pip \
    git curl wget nano vim ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Update pip
RUN python -m pip install --upgrade pip

# --- Install JAX (GPU) ---
RUN pip install \
    jax==0.4.29 \
    jaxlib==0.4.29+cuda12.cudnn91 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# --- Create workspace directory ---
WORKDIR /workspace

# --- Keep container alive on RunPod ---
# Use ENTRYPOINT, NOT CMD, because RunPod overrides CMD
ENTRYPOINT ["tail", "-f", "/dev/null"]
