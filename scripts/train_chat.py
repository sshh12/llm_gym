from dataclasses import dataclass, field
from typing import List
import logging

import transformers
import torch
import tqdm
import random
import wandb

from llm_gym.agents.mistral import MistralChatAgent

# from llm_gym.envs.math_basic import MathBasicEnv
from llm_gym.envs.math_funcs import MathFuncsEnv
from llm_gym.data_utils import ExampleDataset


@dataclass
class TrainingArguments:
    num_iterations: int = field(
        default=10_000, metadata={"help": "Number of training iterations."}
    )

    num_envs: int = field(
        default=20, metadata={"help": "Target number of environments."}
    )
    num_inference_batch_size: int = field(
        default=1,
        metadata={"help": "Max number of environments to run at the same time"},
    )
    per_step_max_new_tokens: int = field(default=2048)

    per_device_train_batch_size: int = field(default=1)
    per_iteration_train_epochs: int = field(default=10)
    gradient_accumulation_steps: int = field(default=16)
    lr: float = field(
        default=1e-4, metadata={"help": "Learning rate for the optimizer."}
    )


def _sample_and_table(values: List, n: int, step: int):
    if len(values) > n:
        values = random.sample(values, k=n)
    keys = list(values[0].keys())
    return wandb.Table(
        columns=keys + ["step"],
        data=[[str(val[k]) for k in keys] + [step] for val in values],
    )


def sample_policy(training_args, agent) -> List:
    envs = [MathFuncsEnv() for _ in range(training_args.num_envs)]
    for env in envs:
        env.reset()

    def yield_ready_envs(batch_size: int = training_args.num_inference_batch_size):
        while not all(env.done for env in envs):
            ready_envs = []
            for env in envs:
                if env.done:
                    continue
                ready_envs.append(env)
            while len(ready_envs) > 0:
                batch_envs = ready_envs[:batch_size]
                ready_envs = ready_envs[batch_size:]
                yield batch_envs

    with torch.no_grad():
        for batch_envs in tqdm.tqdm(yield_ready_envs(), unit="inference_batch"):
            batch_obs = [agent.encode_chat(env.observe()) for env in batch_envs]
            results = agent.generate(
                agent.batch_inputs(batch_obs),
                max_new_tokens=training_args.per_step_max_new_tokens,
            )
            for i, env in enumerate(batch_envs):
                env.step(results[i])

    return envs


# obs = agent.encode_chat(
#     [
#         {
#             "role": "user",
#             "content": "# Context\nYou are an expert assistant that can run python to help answer questions.\n\n```python\n...\n```\n\n# Prompt\n What is 30707 + 31002? Hint: Use \n```python\na * b``` first and then copy the output of this.",
#         }
#     ]
# )
# agent.decode(
#     agent.generate(
#         agent.batch_inputs([obs]),
#         max_new_tokens=training_args.per_step_max_new_tokens,
#     )[0]
# )


def main(
    training_args: TrainingArguments,
):
    agent = MistralChatAgent.load_from_path()

    wandb.init(project="llm_gym", config=training_args.__dict__)

    for iteration in range(training_args.num_iterations):
        agent.eval()

        envs = sample_policy(training_args, agent)

        args = transformers.TrainingArguments(
            output_dir="/data/llm_gym_train",
            lr_scheduler_type="constant",
            max_grad_norm=1.0,
            remove_unused_columns=False,
            learning_rate=training_args.lr,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            bf16=True,
            num_train_epochs=training_args.per_iteration_train_epochs,
            logging_steps=1,
            report_to=[],
            evaluation_strategy="no",
        )

        examples = []
        raw_examples = []
        for env in envs:
            for ex in env.to_examples():
                raw_examples.append(ex.copy())
                examples.append(agent.encode_chat_with_labels(ex["chat"]))
        train_dataset = ExampleDataset(examples)

        if len(train_dataset) == 0:
            print("!!!! NOT DATA, RESAMPING")
            continue

        agent.train()

        with torch.set_grad_enabled(True):
            trainer = transformers.Trainer(
                model=agent.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=agent.batch_inputs,
            )
            train_results = trainer.train()

        wandb.log(
            {
                "loss": train_results.training_loss,
                "correct_cnt": sum([env.first_correct for env in envs]),
                "correct2_cnt": sum(
                    [env.first_correct or env.second_correct for env in envs]
                ),
                "iteration": iteration,
                "sample_examples": _sample_and_table(raw_examples, 10, step=iteration),
            },
            commit=True,
        )

        del examples
        del train_dataset
        del envs
        del trainer


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((TrainingArguments,))

    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(training_args)
