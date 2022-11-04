from abc import ABC, abstractmethod

import torch as th


class ITrainPredImageLogger(ABC):

    @abstractmethod
    def log_image(self, name: str, preds: th.Tensor, targets: th.Tensor,
                  step: int):
        pass
