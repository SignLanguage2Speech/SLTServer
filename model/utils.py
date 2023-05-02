import torch
from time import time

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