# dataset
# -- utils.py

import json
import jsonlines
import io
import torch.distributed as dist

# jsonl
def is_rank_0()->bool:#判断当前进程是否为主进程（rank0）
    return not dist.is_initialized() or dist.get_rank()==0

def _make_r_io_base(f, mode:str):# 输入是io。IOBase实例或者文件路径
    if not isinstance(f, io.IOBase):
        f = open(f, mode = mode,encoding ='utf-8')
    return f

def jload(f, mode='r'):
    f = _make_r_io_base(f,mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jsonl_load(file):
    ret = []
    with jsonlines.open(file) as lines:
        for line in lines:
            ret.append(line)
    return ret

