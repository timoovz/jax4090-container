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

# JAX with CUDA 12 (same command you used manually)
RUN python -m pip install -U "jax[cuda12]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Minimal SSH entrypoint that makes TCP SSH work when ROOT_AUTHORIZED_KEY is set
RUN bash -lc 'cat <<"EOF" > /entrypoint-sshd.sh
#!/usr/bin/env bash
set -e

# Ensure sshd runtime dir exists
mkdir -p /var/run/sshd

# Root SSH dir
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Optional: public key from env (set this in RunPod template)
if [ -n "${ROOT_AUTHORIZED_KEY:-}" ]; then
  echo "${ROOT_AUTHORIZED_KEY}" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# Make sure sshd allows root login and only key auth
if grep -q "^PermitRootLogin" /etc/ssh/sshd_config; then
  sed -i "s/^PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
else
  echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
fi

if grep -q "^PasswordAuthentication" /etc/ssh/sshd_config; then
  sed -i "s/^PasswordAuthentication.*/PasswordAuthentication no/" /etc/ssh/sshd_config
else
  echo "PasswordAuthentication no" >> /etc/ssh/sshd_config
fi

exec /usr/sbin/sshd -D
EOF
chmod +x /entrypoint-sshd.sh'

WORKDIR /workspace

# Expose SSH port (RunPod maps this to the external TCP port)
EXPOSE 22

ENTRYPOINT ["/entrypoint-sshd.sh"]
