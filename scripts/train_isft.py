from dataclasses import dataclass, field
from typing import List
import logging

from trl import IterativeSFTTrainer
import transformers
import torch
import tqdm
import random
import wandb

from llm_gym.agents.mistral import MistralQLoRAChatAgent

# from llm_gym.envs.math_basic import MathBasicEnv
from llm_gym.envs.math_funcs import PythonMathHintsEnv
from llm_gym.envs.env_utils import aggregate_stats


@dataclass
class TrainArguments:
    num_iterations: int = field(
        default=10_000, metadata={"help": "Number of training iterations."}
    )

    num_envs: int = field(
        default=16, metadata={"help": "Target number of environments."}
    )
    inference_batch_size: int = field(
        default=1,
        metadata={"help": "Max number of environments to run at the same time"},
    )
    inference_max_new_tokens: int = field(default=2048)

    per_device_train_batch_size: int = field(default=1)
    per_iteration_train_epochs: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    lr: float = field(
        default=1e-4, metadata={"help": "Learning rate for the optimizer."}
    )


def sample_policy(training_args, agent) -> List:
    envs = [PythonMathHintsEnv() for _ in range(training_args.num_envs)]
    for env in envs:
        env.reset()

    def yield_ready_envs(batch_size: int = training_args.inference_batch_size):
        while not all(env.done for env in envs):
            ready_envs = []
            for env in envs:
                if env.is_done() or not env.is_ready():
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
                max_new_tokens=training_args.inference_max_new_tokens,
            )
            for i, env in enumerate(batch_envs):
                env.step(results[i])

    stats = aggregate_stats([env.get_stats() for env in envs])

    return envs, stats


def main(
    training_args: TrainArguments,
):
    agent = MistralQLoRAChatAgent.load_from_path()

    wandb.init(project="llm_gym2", config=training_args.__dict__)

    trainer_config = transformers.TrainingArguments(
        output_dir="/data/llm_gym_train",
        lr_scheduler_type="constant",
        max_grad_norm=1.0,
        remove_unused_columns=False,
        learning_rate=training_args.lr,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        bf16=True,
        tf32=True,
        num_train_epochs=training_args.per_iteration_train_epochs,
        logging_steps=1,
        report_to=[],
        evaluation_strategy="no",
    )

    trainer = IterativeSFTTrainer(
        model=agent.model,
        tokenizer=agent.tokenizer,
        args=trainer_config,
        data_collator=agent.batch_inputs,
    )

    for iteration in range(training_args.num_iterations):
        trainer.model.eval()

        envs, stats = sample_policy(training_args, agent)

        examples = []
        for env in envs:
            for ex in env.to_examples():
                if ex["reward"] > 0.0:
                    examples.append(agent.encode_chat_with_labels(ex["chat"]))

        trainer.step(
            input_ids=[ex["input_ids"] for ex in examples],
            attention_mask=[ex["attention_mask"] for ex in examples],
            labels=[ex["labels"] for ex in examples],
        )
        print("stats", stats)

        del examples

        wandb.log(
            {"iteration": iteration, **stats},
            commit=True,
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((TrainArguments,))

    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(training_args)
