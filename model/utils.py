import torch

def list2tensor(lst):
    return torch.Tensor(lst)

def tensor2list(tns):
    return tns.tolist()

def videobuffer2tensor(bin_bytes):
    return torch.frombuffer(bin_bytes, dtype=torch.uint8)