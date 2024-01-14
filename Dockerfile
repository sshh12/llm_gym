FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /app

RUN apt-get update && apt-get install -y git curl nano wget unzip rsync jq

RUN git clone https://github.com/sshh12/llm_gym \
        && cd llm_gym \
        && pip install -r requirements.txt \
        && pip install -e .

RUN pip install flash-attn --no-build-isolation

CMD bash