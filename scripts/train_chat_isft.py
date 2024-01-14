from dataclasses import dataclass, field
from typing import List
import logging

from trl import IterativeSFTTrainer
import transformers
import torch
import tqdm
import wandb

from llm_gym.agents import CHAT_MODEL_NAME_TO_CLASS
from llm_gym.envs.basic_math_envs import PythonMathHintsEnv
from llm_gym.envs.meta_math_qa import MetaMathGPTEvalHintsEnv
from llm_gym.envs.env_utils import aggregate_stats


@dataclass
class TrainArguments:
    num_iterations: int = field(
        default=100_000, metadata={"help": "Number of training iterations."}
    )
    num_envs: int = field(
        default=16, metadata={"help": "Target number of environments."}
    )

    inference_batch_size: int = field(
        default=1,
        metadata={"help": "Max number of environments to run at the same time"},
    )
    per_device_train_batch_size: int = field(default=1)
    per_iteration_train_epochs: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    inference_max_new_tokens: int = field(default=2048)

    bf16: bool = field(default=False, metadata={"help": "NVIDIA BF16 mode."})
    tf32: bool = field(default=False, metadata={"help": "NVIDIA TF32 mode."})

    lr: float = field(
        default=1e-4, metadata={"help": "Learning rate for the optimizer."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    wandb_project: str = field(default="llm_gym")


@dataclass
class ModelArguments:
    model_cls: str = field(
        default="MistralQLoRAChatAgent",
        metadata={"help": "Model class to use for chat."},
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Model name or path to load."},
    )


def sample_policy(training_args, agent) -> List:
    envs = [
        MetaMathGPTEvalHintsEnv(max_attempts=2) for _ in range(training_args.num_envs)
    ]
    for env in envs:
        env.reset()

    def yield_ready_envs(batch_size: int = training_args.inference_batch_size):
        while not all(env.done for env in envs):
            ready_envs = []
            for env in envs:
                if env.is_done() or not env.is_ready():
                    continue
                ready_envs.append(env)
            batch_envs = ready_envs[:batch_size]
            yield batch_envs

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
    model_args: ModelArguments,
):
    agent_cls = CHAT_MODEL_NAME_TO_CLASS[model_args.model_cls]
    agent_args = dict()
    if model_args.model_name_or_path is not None:
        agent_args["model_name_or_path"] = model_args.model_name_or_path
    agent = agent_cls.load_from_path(**agent_args)

    trainer_config = transformers.TrainingArguments(
        output_dir="/tmp/llm_gym_output",
        lr_scheduler_type="constant",
        max_grad_norm=training_args.max_grad_norm,
        remove_unused_columns=False,
        learning_rate=training_args.lr,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        bf16=training_args.bf16,
        tf32=training_args.tf32,
        num_train_epochs=training_args.per_iteration_train_epochs,
        logging_steps=1,
        report_to=[],
        evaluation_strategy="no",
        max_steps=training_args.num_iterations,
    )

    # trainer = IterativeSFTTrainer(
    #     model=agent.model,
    #     tokenizer=agent.tokenizer,
    #     args=trainer_config,
    #     data_collator=agent.batch_inputs,
    # )

    # wandb.init(
    #     project=training_args.wandb_project,
    #     config={**training_args.__dict__, **model_args.__dict__},
    # )

    for iteration in range(training_args.num_iterations):
        # trainer.model.eval()
        torch.cuda.empty_cache()

        with torch.no_grad():
            envs, stats = sample_policy(training_args, agent)

        examples = []
        for env in envs:
            for ex in env.to_examples():
                if ex.reward > 0.0:
                    examples.append(agent.encode_chat_with_labels(ex.chat))

        trainer.step(
            input_ids=[ex["input_ids"] for ex in examples],
            attention_mask=[ex["attention_mask"] for ex in examples],
            labels=[ex["labels"] for ex in examples],
        )
        print("stats", stats)

        del examples

        # wandb.log(
        #     {"iteration": iteration, **stats},
        #     commit=True,
        # )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((TrainArguments, ModelArguments))

    training_args, model_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    main(training_args, model_args)
