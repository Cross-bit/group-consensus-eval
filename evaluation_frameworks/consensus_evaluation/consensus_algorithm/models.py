
from dataclasses import dataclass


@dataclass
class Vote:
    id: int
    value: int

    def __eq__(self, other):
        return isinstance(other, Vote) and self.id == other.id

    def __hash__(self):
        return hash(self.id)