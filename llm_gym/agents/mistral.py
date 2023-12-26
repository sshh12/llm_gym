from typing import Optional, Dict, List
from dataclasses import dataclass

from torch.distributions.categorical import Categorical
from transformers import AutoTokenizer, MistralForCausalLM, LlamaTokenizer
import torch
import torch.nn as nn

from llm_gym.model_utils import layer_init, fix_tokenizer, load_model_qlora, left_pad


class DiagLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(DiagLinear, self).__init__(in_features, out_features)
        torch.nn.init.eye_(self.weight)
        torch.nn.init.zeros_(self.bias)


class MistralAgent(nn.Module):
    def __init__(
        self,
        model: MistralForCausalLM,
        value_model: MistralForCausalLM,
        tokenizer: LlamaTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.llm = model.model.model
        self.value_llm = value_model.model.model
        self.lm_head = model.lm_head
        self.value_head = nn.Linear(4096, 1).to(self.value_llm.device)

    def get_value(self, x: Dict):
        outputs = self.value_llm(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False,
        )
        hidden_states = outputs[0]
        return self.value_head(hidden_states[:, -1])

    def get_action_and_value(
        self, x: Dict, action: Optional[List[int]] = None, temperature: float = 1.0
    ):
        outputs = self.llm(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=False,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -1]) / temperature
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.get_value(x),
        )

    def batch_inputs(self, xs: List[Dict]) -> Dict:
        return {
            "input_ids": left_pad(
                [x["input_ids"] for x in xs],
                self.tokenizer.pad_token_id,
            ).to(self.llm.device),
            "attention_mask": left_pad(
                [x["attention_mask"] for x in xs],
                0,
            ).to(self.llm.device),
        }

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

        value_model = load_model_qlora(MistralForCausalLM, model_name_or_path)
        model = load_model_qlora(MistralForCausalLM, model_name_or_path)
        agent = cls(model, value_model, tokenizer)
        return agent
