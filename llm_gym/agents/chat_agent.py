from typing import Dict, List
import abc


class BaseChatAgent(abc.ABC):
    @abc.abstractmethod
    def generate(self, x: Dict, **kwargs) -> List[str]:
        pass

    @abc.abstractmethod
    def batch_inputs(self, xs: List[Dict]) -> Dict:
        pass

    @abc.abstractmethod
    def encode_chat(self, chat: List[Dict]) -> Dict:
        pass

    @abc.abstractmethod
    def encode_chat_with_labels(
        self,
        chat: List[Dict],
    ) -> Dict:
        pass

    def eval(self):
        pass

    def train(self):
        pass

    @abc.abstractclassmethod
    def load_from_path(cls, *args, **kwargs) -> "BaseChatAgent":
        pass
