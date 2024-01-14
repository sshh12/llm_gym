from typing import Dict, List
from abc import ABC, abstractmethod


class BaseChatAgent(ABC):
    @abstractmethod
    def generate(self, x: Dict, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def batch_inputs(self, features: List) -> Dict:
        pass

    @abstractmethod
    def encode_chat(self, chat: List[Dict]) -> Dict:
        pass

    @abstractmethod
    def encode_chat_with_labels(
        self,
        chat: List[Dict],
    ) -> Dict:
        pass

    @classmethod
    def load_from_path(cls, *args, **kwargs) -> "BaseChatAgent":
        pass
