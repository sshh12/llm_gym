#!/bin/bash
# curl -o- https://raw.githubusercontent.com/sshh12/llm_gym/main/scripts/vastai_setup.sh | bash
apt-get update && apt-get install -y git curl nano wget unzip rsync jq

git clone https://github.com/sshh12/llm_gym \
        && cd llm_gym \
        && pip install -r requirements.txt \
        && pip install -e .

pip install flash-attn --no-build-isolation
pip install wandb