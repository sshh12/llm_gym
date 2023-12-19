from typing import List, Any

import torch
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


def make_model_lora(model: Any) -> Any:
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
