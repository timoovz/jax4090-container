FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Core tools + SSH + Python 3.11
RUN apt-get update && apt-get install -y \
    openssh-server \
    python3.11 python3.11-dev python3.11-distutils python3.11-venv python3-pip \
    git curl wget nano vim ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# JAX with CUDA 12 – same style as you did manually in the pod
RUN python -m pip install -U "jax[cuda12]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# SSH setup: dirs + allow root login with pubkey over TCP
RUN mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh

RUN bash -lc '\
  if grep -q "^PermitRootLogin" /etc/ssh/sshd_config; then \
    sed -i "s/^PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config; \
  else \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config; \
  fi; \
  if grep -q "^PubkeyAuthentication" /etc/ssh/sshd_config; then \
    sed -i "s/^PubkeyAuthentication.*/PubkeyAuthentication yes/" /etc/ssh/sshd_config; \
  else \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config; \
  fi; \
  if grep -q "^PasswordAuthentication" /etc/ssh/sshd_config; then \
    sed -i "s/^PasswordAuthentication.*/PasswordAuthentication no/" /etc/ssh/sshd_config; \
  else \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config; \
  fi'

WORKDIR /workspace

# Expose SSH port (RunPod will map this to some external port like 14979)
EXPOSE 22

# Keep container alive via sshd
CMD ["/usr/sbin/sshd", "-D"]
