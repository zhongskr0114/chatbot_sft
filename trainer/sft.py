# trainer
# -- sft.py
import time
from typing import Optional,Any

import torch
from attr.validators import max_len
from torch import nn
from torch.optim import Optimizer
from torch.utils._pytree import tree_map
import tqdm
import wandb
from torch.utils.data import DataLoader
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention

from .base import Trainer


def to_device(x:Any, device:torch.device):
    def _to(t:Any):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return t
    return tree_map(_to, x)

class SFTTrainer(Trainer):
    def __init__(self,  model:nn.Module,
                 optimizer:Optimizer, lr_scheduler,
                 device = 'cuda',batch_size:int = 2,max_epochs:int=2)->None:
        super(SFTTrainer, self).__init__(max_epochs,model,optimizer)
        self.scheduler = lr_scheduler
        self.batch_size = batch_size
        self.device = torch.device(device)
    def _before_fit(self,train_dataloader:DataLoader,
                    eval_dataloader: Optional[DataLoader] = None,
                    logger: Optional = None,
                    use_wandb :bool = False):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.logger = logger
        #self.use_wandb = use_wandb
        self.total_loss = 0
        self.no_epoch_bar = True
        self.step_bar = tqdm.trange(len(self.train_dataloader)//self.batch_size*self.max_epochs,
            desc='steps')


    def _train(self, epoch):
        self.model.train()
        for batch_id,batch in enumerate(self.train_dataloader):
            batch = to_device(batch, self.device)
            outputs = self.model(batch['input_ids'],
                                 attention_mask = batch['attention_mask'],
                                 labels = batch['labels']
                                 )
            loss = outputs.loss
            loss.backward()
            self.total_loss+= loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.logger.info({'loss':self.total_loss,
                          'lr':self.scheduler.get_last_lr()[0],
                          'epoch':epoch,
                          'batch_id':batch_id})
            self.step_bar.update()

    def _eval(self, epoch:int):
        if self.eval_dataloader is not None:
            self.model.eval()
            with torch.no_grad():
                loss_sum,num_seen=0,0
                for batch in self.eval_dataloader:
                    batch = to_device(batch, self.device)
                    outputs = self.model(batch['input_ids'],
                                         attention_mask = batch['attention_mask'],
                                         labels = batch['labels'])
                    loss = outputs.loss
                    loss_sum += loss.item()
                    num_seen += batch['input_ids'].size(0)
                loss_mean = loss_sum/num_seen
                self.logger.info(f'eval loss:{loss_mean}')

