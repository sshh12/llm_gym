from typing import Optional, List, Dict
from dataclasses import dataclass, field
import logging
import tqdm
import math
import random
import itertools

import torch.nn.functional as F
import torch.optim as optim
import transformers
import wandb
import torch

from llm_gym.agents.mistral import MistralAgent
from llm_gym.gyms.basic_math import BasicMathGym


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


@dataclass
class TrainingArguments:
    num_iterations: int = field(
        default=10_000, metadata={"help": "Number of training iterations."}
    )
    lr: float = field(
        default=1e-4, metadata={"help": "Learning rate for the optimizer."}
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
        default=0.001, metadata={"help": "Entropy coefficient for the loss."}
    )
    vf_coef: float = field(
        default=0.5, metadata={"help": "Value function coefficient for the loss."}
    )
    max_grad_norm: float = field(
        default=0.5, metadata={"help": "Maximum gradient norm for clipping."}
    )


def lr_warmup_cosine(optimizer, warmup_steps, total_steps, max_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max_lr * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(
    training_args: TrainingArguments,
):
    device = torch.device("cuda")
    agent = MistralAgent.load_from_path(device=device)

    with open("params.txt", "w") as f:
        for name, param in agent.named_parameters():
            f.write(f"{name} {param.shape} {param.requires_grad}\n")

    optimizer = optim.Adam(agent.parameters(), lr=training_args.lr, eps=1e-5)
    # lr_scheduler = lr_warmup_cosine(
    #     optimizer,
    #     warmup_steps=10,
    #     total_steps=training_args.num_iterations,
    #     max_lr=training_args.lr,
    # )

    num_envs = 16
    num_epochs = 1
    num_accumulation = 16

    # lr schedule
    # temp schedule

    wandb.init(project="llm_gym")

    for iteration in range(training_args.num_iterations):
        print(f"Iteration {iteration} of {training_args.num_iterations}")

        # if args.anneal_lr:
        #     frac = 1.0 - (iteration - 1.0) / args.num_iterations
        #     lrnow = frac * args.learning_rate
        #     optimizer.param_groups[0]["lr"] = lrnow

        actions = {}
        logprobs = {}
        rewards = {}
        values = {}
        dones = {}
        obs = {}
        step = {}

        envs = [BasicMathGym(tokenizer=agent.tokenizer) for _ in range(num_envs)]
        for env_idx, env in enumerate(envs):
            obs[(env_idx, 0)] = env.reset()
            step[env_idx] = 0
            dones[env_idx] = False

        def yield_ready_envs(batch_size: int = num_envs):
            while not all(dones.values()):
                ready_env_idxs = []
                for env_idx, _ in enumerate(envs):
                    if dones[env_idx]:
                        continue
                    ready_env_idxs.append(env_idx)
                while len(ready_env_idxs) > 0:
                    batch_idxs = ready_env_idxs[:batch_size]
                    ready_env_idxs = ready_env_idxs[batch_size:]
                    yield batch_idxs, [envs[ei] for ei in batch_idxs]

        with torch.no_grad():
            for batch_env_idxs, batch_envs in tqdm.tqdm(
                yield_ready_envs(), unit="inference_batch"
            ):
                batch_obs = [obs[(ei, step[ei])] for ei in batch_env_idxs]
                action, logprob, _, value = agent.get_action_and_value(
                    agent.batch_inputs(batch_obs)
                )
                for i, env_idx in enumerate(batch_env_idxs):
                    cur_step = step[env_idx]
                    values[(env_idx, cur_step)] = value[i]
                    actions[(env_idx, cur_step)] = action[i]
                    logprobs[(env_idx, cur_step)] = logprob[i]

                    next_obs, reward, done = batch_envs[i].step(
                        action=int(action[i].cpu().numpy())
                    )
                    obs[(env_idx, cur_step + 1)] = next_obs
                    rewards[(env_idx, cur_step)] = reward
                    dones[env_idx] = done
                    step[env_idx] += 1

        # import IPython

        # IPython.embed()

        advantages = {}

        for env_idx in range(len(envs)):
            lastgaelam = 0
            env_num_steps = step[env_idx]
            for t in reversed(range(env_num_steps)):
                if t == env_num_steps - 1:
                    nextnonterminal = 0.0
                    nextvalues = 0.0
                else:
                    nextnonterminal = 1.0
                    nextvalues = values[(env_idx, t + 1)]
                delta = (
                    rewards[(env_idx, t)]
                    + training_args.gamma * nextvalues * nextnonterminal
                    - values[(env_idx, t)]
                )
                advantages[(env_idx, t)] = lastgaelam = (
                    delta
                    + training_args.gamma
                    * training_args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )

        idx_to_env_step = {i: env_step for i, env_step in enumerate(actions.keys())}
        b_logprobs = torch.Tensor(
            [logprobs[env_step] for env_step in idx_to_env_step.values()]
        ).to(device)
        del logprobs
        b_actions = torch.IntTensor(
            [actions[env_step] for env_step in idx_to_env_step.values()]
        ).to(device)
        del actions
        b_advantages = torch.Tensor(
            [advantages[env_step] for env_step in idx_to_env_step.values()]
        ).to(device)
        del advantages
        b_values = torch.Tensor(
            [values[env_step] for env_step in idx_to_env_step.values()]
        ).to(device)
        del values
        b_returns = b_advantages + b_values

        for _ in range(num_epochs):
            idxs = list(range(len(idx_to_env_step)))

            random.shuffle(idxs)
            idxs = idxs[: num_accumulation * 64]

            # idxs = idxs[: len(idxs) - (len(idxs) % num_accumulation)]

            batches = list(batched(idxs, 1))

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                for bi, batch_idxs in enumerate(
                    tqdm.tqdm(batches, unit="training_batch")
                ):
                    batch_obs = [
                        obs[(idx_to_env_step[i][0], idx_to_env_step[i][1])]
                        for i in batch_idxs
                    ]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        agent.batch_inputs(batch_obs),
                        b_actions.long()[batch_idxs],
                    )
                    logratio = newlogprob - b_logprobs[batch_idxs]
                    ratio = logratio.exp()

                    # with torch.no_grad():
                    #     # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    #     # old_approx_kl = (-logratio).mean()
                    #     # approx_kl = ((ratio - 1) - logratio).mean()
                    #     clipfracs += [
                    #         ((ratio - 1.0).abs() > training_args.clip_coef)
                    #         .float()
                    #         .mean()
                    #         .item()
                    #     ]

                    mb_advantages = b_advantages[batch_idxs]

                    policy_clip_coef = training_args.clip_coef

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - policy_clip_coef, 1 + policy_clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    pg_loss = pg_loss * (min(iteration, 20) / 20)

                    # Value loss
                    newvalue = newvalue.view(-1)

                    value_coef_func = (
                        lambda x: 10000
                        if x == 0
                        else (1.0 if x >= 10 else 10000 - (9999 / 9) * x)
                    )
                    value_clip_coef = training_args.clip_coef * value_coef_func(
                        iteration
                    )
                    if False:
                        v_loss_unclipped = (newvalue - b_returns[batch_idxs]) ** 2
                        v_clipped = b_values[batch_idxs] + torch.clamp(
                            newvalue - b_values[batch_idxs],
                            -value_clip_coef,
                            value_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[batch_idxs]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[batch_idxs]) ** 2).mean()

                    entropy_loss = entropy.mean()

                    if iteration < 5:
                        pg_loss *= 0.0
                        entropy_loss = entropy.mean() * 0.0

                    loss = (
                        pg_loss
                        - entropy_loss * training_args.ent_coef
                        + v_loss * training_args.vf_coef
                    )
                    loss = loss / num_accumulation
                    loss.backward()

                    if (bi + 1) % num_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(
                            agent.parameters(), training_args.max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad()

            wandb.log(
                {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "correct_cnt": sum([env.correct for env in envs]),
                    "value_loss": v_loss.item(),
                    "policy_loss": pg_loss.item(),
                    "entropy_loss": entropy_loss.item(),
                    "loss": loss.item(),
                },
                commit=True,
            )
        # lr_scheduler.step()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((TrainingArguments,))

    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(training_args)
