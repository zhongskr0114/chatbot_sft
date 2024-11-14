# trainer:
# -- base.py
from abc import abstractmethod,ABC #(Abstract Base Class,相当于提供了一个接口（无法实例化）)
import torch
from torch import nn
from torch.optim import Optimizer
import tqdm

class Trainer(ABC):
    def __init__(self, max_epochs:int , model:nn.Module, optimizer:Optimizer):
        super(Trainer, self).__init__()
        self.max_epochs = max_epochs
        self.model = model
        self.optimizer = optimizer
    @abstractmethod
    def _train(self, epoch):
        raise NotImplementedError()

    @abstractmethod
    def _eval(self, epoch):
        raise NotImplementedError()


    def _before_fit(self):
        print('父类_before_fit')
        self.no_epoch_bar = False

    def fit(self, *args, **kwards):
        self._before_fit(*args, **kwards)
        for epoch in tqdm.trange(self.max_epochs, desc='epochs', disable= self.no_epoch_bar):
            self._train(epoch)
            self._eval(epoch)

 