import torch
    
class Logger:
    debug = False
    mode2flag = {
            0: "INFO",
            1: "STARTING",
            2: "FINISHED",
            -1:"ERROR",
        }

    @staticmethod
    def toggle_dbg(debug=True):
        Logger.debug = debug

    @staticmethod
    def log_dbg(self, *args, mode=0):
        if Logger.debug:
            print(f"DEBUG @ {Logger.mode2flag[mode]} ::", *args)

def list2tensor(lst):
    return torch.Tensor(lst)

def tensor2list(tns):
    return tns.tolist()

def videobuffer2tensor(bin_bytes):
    return torch.frombuffer(bin_bytes, dtype=torch.uint8)