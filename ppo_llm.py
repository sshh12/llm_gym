from typing import List
from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
)
import numpy as np
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch
import torch.nn as nn
import random


def _find_all_linear_names(model) -> List[str]:
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def _make_model_lora(model):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=_find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.to(torch.bfloat16)
    model = get_peft_model(model, lora_config)
    return model


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def fix_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.unk_token
    if tokenizer.cls_token is None:
        tokenizer.cls_token = tokenizer.unk_token
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.unk_token


class Agent(nn.Module):
    def __init__(self, model: MistralForCausalLM):
        super().__init__()
        self.llm = model.model.model
        self.lm_head = model.lm_head
        self.critic_head = layer_init(
            nn.Linear(4096, 1),
            std=1,
        )

    def get_value(self, x, done):
        outputs = self.llm(
            input_ids=x,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False,
        )
        hidden_states = outputs[0]
        return self.critic_head(hidden_states[:, -1].float()).to(self.llm.dtype)

    def get_action_and_value(self, x, done, action=None):
        outputs = self.llm(
            input_ids=x,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=False,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -1])
        # print("logits", logits)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_head(hidden_states[:, -1].float()).to(self.llm.dtype),
        )


class MathEnv:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def reset(self):
        self.a = random.randint(0, 100_000)
        self.b = random.randint(0, 100_000)
        self.prompt = f"What is {self.a} + {self.b}?"
        self.state = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": self.prompt},
            ],
            return_tensors="pt",
            padding=True,
            max_length=25,
        )[0]
        self.done = False
        self.correct = False
        return self.state

    def step(self, action: int):
        reward = 0
        if not self.done and action == self.tokenizer.eos_token_id:
            resp = self.tokenizer.decode(self.state)
            correct = str(self.a + self.b) in resp.replace(",", "")
            print("done!", resp, "CORRECT" if correct else "INCORRECT")
            reward = 1.0 if correct else -1.0
            self.correct = correct
            self.done = True
        if not self.done:
            self.state = torch.concat([self.state, torch.IntTensor([action])])
        else:
            self.state = torch.concat(
                [self.state, torch.IntTensor([self.tokenizer.eos_token_id])]
            )
        return self.state, reward, int(self.done)


class MultiEnv:
    def __init__(self, envs):
        self.envs = envs

    def reset(self):
        return torch.vstack([env.reset() for env in self.envs])

    def step(self, actions):
        next_step = [env.step(action) for env, action in zip(self.envs, actions)]
        next_obs = torch.vstack([step[0] for step in next_step])
        rewards = torch.IntTensor([step[1] for step in next_step])
        dones = torch.IntTensor([step[2] for step in next_step])
        return next_obs, rewards, dones

    def correct_pct(self):
        return np.mean([env.correct for env in self.envs])


def main(
    model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
    learning_rate: float = 2.5e-4,
    num_steps: int = 50,
    num_envs: int = 10,
    num_iterations: int = 10_000,
    num_minibatches: int = 1,
    update_epochs: int = 1,
    anneal_lr: bool = True,
    norm_adv: bool = True,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.1,
    clip_vloss: bool = True,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float = None,
):
    import wandb

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    fix_tokenizer(tokenizer)
    model = MistralForCausalLM.from_pretrained(model_name_or_path)
    model = _make_model_lora(model)

    device = torch.device("cuda")

    agent = Agent(model).to(device)

    with open("params.txt", "w") as f:
        for name, param in agent.named_parameters():
            f.write(f"{name} {param.shape} {param.requires_grad}\n")

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    wandb.init(project="llm_gym")
    global_step = 0

    for iteration in range(num_iterations):
        print(f"iteration {iteration} of {num_iterations}")

        envs = MultiEnv([MathEnv(tokenizer) for _ in range(num_envs)])
        # obs = torch.zeros((num_steps, num_envs, num_steps)).to(device)
        actions = torch.zeros((num_steps, num_envs, 1)).int().to(device)
        logprobs = torch.zeros((num_steps, num_envs)).to(device)
        rewards = torch.zeros((num_steps, num_envs)).to(device)
        dones = torch.zeros((num_steps, num_envs)).to(device)
        values = torch.zeros((num_steps, num_envs)).to(device)
        next_obs = envs.reset().to(device)
        next_done = torch.zeros(num_envs).to(device)

        obs = []

        if anneal_lr:
            if iteration < 10:
                lrnow = learning_rate * ((iteration + 1) / 10)
            else:
                frac = 1.0 - (iteration / num_iterations)
                lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += num_envs

            # import IPython

            # IPython.embed()

            # obs[step] = next_obs
            # obs = step x envs x tokens
            obs.append(next_obs)
            dones[step] = next_done

            with torch.no_grad():
                for e in range(num_envs):
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs[[e]], next_done[[e]]
                    )
                    values[step, e] = value.flatten()

                    actions[step, e] = action
                    logprobs[step, e] = logprob

            next_obs, reward, next_done = envs.step(actions[step][:, 0].cpu().numpy())
            rewards[step] = reward.to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

        # print("steps complete", step)

        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 1))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        batch_size = int(num_envs * num_steps)
        assert num_envs % num_minibatches == 0
        envsperbatch = num_envs // num_minibatches
        envinds = torch.arange(num_envs)
        # flatinds = torch.arange(num_envs * num_steps)
        clipfracs = []
        for epoch in range(update_epochs):
            # print(f"epoch {epoch} of {update_epochs}")
            # np.random.shuffle(envinds)
            idxs = list(range(1, b_actions.shape[0]))
            import tqdm

            np.random.shuffle(idxs)
            for i in tqdm.tqdm(idxs):
                # print("batch", i)
                # end = start + envsperbatch
                # mbenvinds = envinds[start:end]
                mb_inds = [[i]]  # be really careful about the index

                # print(b_obs[mb_inds][:, :i])
                # print(b_dones[mb_inds])
                # print(b_actions.long()[mb_inds])
                env_idx = i % num_envs
                step_idx = i // num_envs

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs[step_idx][env_idx].unsqueeze(0),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv and len(mb_advantages) > 3:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # print("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # print("losses/value_loss", v_loss.item(), global_step)
        # print("losses/policy_loss", pg_loss.item(), global_step)
        # print("losses/entropy", entropy_loss.item(), global_step)
        # print("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # print("losses/approx_kl", approx_kl.item(), global_step)
        # print("losses/clipfrac", np.mean(clipfracs), global_step)
        # print("losses/explained_variance", explained_var, global_step)
        wandb.log(
            {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "correct": envs.correct_pct(),
            },
            step=iteration,
        )


if __name__ == "__main__":
    main()
