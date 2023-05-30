import torch
from time import time
from model.Sign2Text.Sign2Text.Sign2Text import Sign2Text
from model.Sign2Text.configs.Sign2Text_config import Sign2Text_cfg
from model.Sign2Text.configs.VisualEncoderConfig import cfg as VisualEncoder_cfg
import os

class Logger:
    debug = False
    measure_time = False
    mode2flag = {
            0: "INFO",
            1: "STARTING",
            2: "FINISHED",
           -1: "ERROR",
        }

    @staticmethod
    def log_dbg(*args, mode=0, prev_start_time=None):
        if Logger.debug:
            start_time = None
            msg_string = f"DEBUG @ {Logger.mode2flag[mode]}"
            if Logger.measure_time:
                if mode==1:
                    start_time = time()
                elif mode==2 and prev_start_time is not None:
                    end_time = time()
                    execution_time = end_time - prev_start_time
                    msg_string += f" after {execution_time:.2f} s"
            print(msg_string, "::", *args)
            return start_time

def list2tensor(lst):
    return torch.Tensor(lst)

def tensor2list(tns):
    return tns.tolist()

def videobuffer2tensor(bin_bytes):
    return torch.frombuffer(bin_bytes, dtype=torch.uint8)

def load_s2t_model(s2t_checkpoint_path, mbart_model_path, vocab_path, device):
    s2t_config = Sign2Text_cfg()
    s2t_config.mbart_path = mbart_model_path
    ve_config = VisualEncoder_cfg(vocab_path)
    ve_config.checkpoint_path = None
    model = Sign2Text(s2t_config, ve_config)
    checkpoint = torch.load(s2t_checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.requires_grad_(False)
    model.eval()
    return model.to(device)