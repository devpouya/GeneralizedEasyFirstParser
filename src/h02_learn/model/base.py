import copy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from utils import constants
from utils import utils


class BaseParser(nn.Module, ABC):
    # pylint: disable=abstract-method
    name = 'base'

    def __init__(self):
        super().__init__()

        self.best_state_dict = self.state_dict()

    def set_best(self):
        with torch.no_grad():
            state_dict = {k: v.detach().cpu() for k, v in self.state_dict().items()}
            self.best_state_dict = copy.deepcopy(state_dict)
            #self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        with torch.no_grad():
            state_dict = {k: v.to(device=constants.device).detach()
                           for k, v in self.best_state_dict.items()}
            self.load_state_dict(state_dict)
            #self.load_state_dict(self.best_state_dict)
        # torch.cuda.empty_cache()

    def save(self, path):
        utils.mkdir(path)
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    @abstractmethod
    def get_args(self):
        pass

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        del checkpoints
        return model

    @classmethod
    def load_checkpoint(cls, path):
        fname = cls.get_name(path)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path):
        return '%s/model.tch' % (path)
