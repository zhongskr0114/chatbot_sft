# train_sft.py
import argparse
from dataclasses import dataclass
import torch
import transformers
from typing import Callable, Dict, List, Sequence

from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention


@dataclass
class CollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict])->Dict[str, torch.Tensor]:
        input_ids = [ins['input_ids'] for ins in instances]
        labels = [ins['labels'] for ins in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                padding_value=-100)
        return dict(input_ids = input_ids, labels=labels,
                    attention_mask = input_ids.ne(self.tokenizer.pad_token_id))

from dataset.sft_dataset import SupervisedDataset
from trainer.sft import SFTTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers.trainer import get_scheduler
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import math

def train(args):
    # chatGLM
    # ~/.cache
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain,
        trust_remote_code=True, cache_dir=args.cache)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.pretrain,
        trust_remote_code=True, cache_dir=args.cache)

    optim = Adam(model.parameters(), lr=args.lr)
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
        data_path=args.dataset,
        max_len=args.max_len
    )
    eval_dataset = None
    data_collator = CollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
        collate_fn=data_collator, pin_memory=True)
    # gpu cpu temp mem(pin mem)
    eval_dataloader = None
    step_per_epoch = len(train_dataloader) // args.batch_size
    max_steps = math.ceil(args.max_epochs * step_per_epoch)
    lr_scheduler = get_scheduler('cosine', optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps
    )
    device = 'cpu'
    trainer = SFTTrainer(model=model, optimizer=optim,
        lr_scheduler=lr_scheduler, max_epochs=args.max_epochs, device=device
    )
    trainer.fit(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
        logger=logging
    )

    state = model.state_dict()
    torch.save(state, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--dataset', type=str, default='ds/alpaca_data_en_52k.jsonl')
    parser.add_argument('--cache', type=str, default='D:/cache')
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()
    train(args)
