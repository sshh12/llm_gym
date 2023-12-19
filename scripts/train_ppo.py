from typing import Optional
from dataclasses import dataclass, field
import logging

import torch.optim as optim
import transformers
import wandb
import torch

from llm_gym.agents.mistral import MistralAgent


@dataclass
class TrainingArguments:
    lr: float = field(
        default=2.5e-4, metadata={"help": "Learning rate for the optimizer."}
    )
    gamma: float = field(
        default=0.99, metadata={"help": "Discount factor for rewards."}
    )
    gae_lambda: float = field(
        default=0.95, metadata={"help": "Lambda parameter for GAE."}
    )
    clip_coef: float = field(
        default=0.1, metadata={"help": "Clipping parameter for PPO."}
    )
    clip_vloss: bool = field(
        default=True, metadata={"help": "Whether to clip the value loss."}
    )
    ent_coef: float = field(
        default=0.01, metadata={"help": "Entropy coefficient for the loss."}
    )
    vf_coef: float = field(
        default=0.5, metadata={"help": "Value function coefficient for the loss."}
    )
    max_grad_norm: float = field(
        default=0.5, metadata={"help": "Maximum gradient norm for clipping."}
    )


def main(
    training_args: TrainingArguments,
):
    device = torch.device("cuda")
    agent = MistralAgent.load_from_path(device=device)

    with open("params.txt", "w") as f:
        for name, param in agent.named_parameters():
            f.write(f"{name} {param.shape} {param.requires_grad}\n")

    optimizer = optim.Adam(agent.parameters(), lr=training_args.lr, eps=1e-5)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((TrainingArguments,))

    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
