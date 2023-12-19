from typing import Optional

from torch.distributions.categorical import Categorical
from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
)
import torch
import torch.nn as nn

from llm_gym.model_utils import layer_init, fix_tokenizer, make_model_lora


class MistralAgent(nn.Module):
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
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_head(hidden_states[:, -1].float()).to(self.llm.dtype),
        )

    @classmethod
    def load_from_path(
        cls,
        model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
        device: Optional[torch.device] = None,
    ) -> "MistralAgent":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        fix_tokenizer(tokenizer)
        model = MistralForCausalLM.from_pretrained(model_name_or_path)
        model = make_model_lora(model)
        agent = cls(model)
        if device is not None:
            agent.to(device)
        return agent
