from typing import List, Dict

from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        super(ExampleDataset, self).__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
