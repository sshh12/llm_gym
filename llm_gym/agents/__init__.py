from llm_gym.agents.phi_agents import Phi2ChatAgent
from llm_gym.agents.mistral_agents import (
    MistralQLoRAChatAgent,
    MistralLoRAChatAgent,
    MistralChatAgent,
)

CHAT_MODEL_CLASSES = [
    Phi2ChatAgent,
    MistralQLoRAChatAgent,
    MistralLoRAChatAgent,
    MistralChatAgent,
]

CHAT_MODEL_NAME_TO_CLASS = {cls.__name__: cls for cls in CHAT_MODEL_CLASSES}
