import torch
from time import time, perf_counter, process_time
from model.Sign2Text.Sign2Text.Sign2Text import Sign2Text
from model.Sign2Text.configs.Sign2Text_config import Sign2Text_cfg
from model.Sign2Text.configs.VisualEncoderConfig import cfg as VisualEncoder_cfg
import os

from datetime import datetime
import pandas as pd

cyan = '\033[96m'
red = '\033[91m'
ylw = '\033[93m'
grn = '\033[92m'
blk = '\033[0m'

kws_to_extract = ["variation", "_input", "test_suite_ID", "repetition_ID"]
""" Wrapper function for monitoring performance. Uses a Pandas DataFrame as a simple database. """
def monitor_perf(func, *args, **kwargs):
    Logger.perf_rep_counter += 1
    extra_kwargs = {}
    for kw in kws_to_extract:
        extra_kwarg = kwargs.get(kw, None)
        if extra_kwarg is not None:
            del kwargs[kw]
        extra_kwargs[kw] = extra_kwarg
        
    start_time, perf_time, proc_time = Logger.log_perf(func.__name__, mode=1, **extra_kwargs)
    res = func(*args, **kwargs)
    Logger.log_perf(func.__name__, mode=2, prev_start_time=start_time, prev_perf=perf_time, prev_proc=proc_time, **extra_kwargs)
    return res



class Logger:
    debug = False
    measure_time = False
    measure_perf = False
    perf_rep_counter = 0
    mode2flag = {
            0: "INFO",
            1: "STARTING",
            2: "FINISHED",
           -1: "ERROR",
        }
    
    """ 
    Test Suite/Repetition IDs:  Certain tests are grouped - this is to identify which group they were part of.
    Function Name:  Automatically retrieved function/method name
    Variation:      Varying different parameters, such as e.g. temporal downsampling, beam_width, etc.
    Input Dims:     Only relevant when a tensor is passed to the monitored function.
    time:           Time delta as measured by Pythons time.time() function.
    perf_counter:   Time delta as measured by Pythons time.perf_counter() function.
    process_time:   Time delta as measured by Pythons time.process_time() function.
    """
    perf_metrics = pd.DataFrame(columns=['Function Name', 'Variation', 'Input Dims', 'time', 'perf_counter', 'process_time', 'Test Suite ID', 'Repetition ID'])
    
    @staticmethod
    def add_to_perf_metrics(func_name, *args, test_suite_ID=None, repetition_ID=None, variation=None, _input=None, _time=None, _perf_counter=None, _process_time=None, **kwargs):
        assert not (_time is None or _perf_counter is None or _process_time == None)
        new_row = pd.DataFrame([{'Function Name': func_name, 
                                  'Variation': variation, 
                                  'Input Dims': _input, 
                                  'time': _time, 
                                  'perf_counter': _perf_counter, 
                                  'process_time': _process_time,
                                  'Repetition ID': repetition_ID if repetition_ID is not None else "None",
                                  'Test Suite ID': test_suite_ID if test_suite_ID is not None else "None"}])

        Logger.perf_metrics = pd.concat([Logger.perf_metrics, new_row], ignore_index=True)
    
    @staticmethod
    def save_perf_metrics(folder_name=None):
        dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        folder_name = folder_name if folder_name is not None else "perf_metrics"
        file_path = os.path.join(os.getcwd(),f"./experiment-data/{folder_name}/perf_metrics_{dt_string}.csv")
        Logger.perf_metrics.to_csv(file_path)

    @staticmethod   
    def reset_perf_metrics():
        Logger.perfmon_step()
        Logger.perf_metrics = Logger.perf_metrics.drop(Logger.perf_metrics.index)
    
    @staticmethod
    def load_perf_metrics(file_name, folder_name=None):
        assert Logger.perf_metrics.empty, "perf_metrics not empty, handle first!"
        folder_name = folder_name if folder_name is not None else "perf_metrics"
        file_name = f"{file_name}.csv" if file_name[-4:] != ".csv" else file_name
        file_path = os.path.join(os.getcwd(),f"./experiment-data/{folder_name}/{file_name}")
        Logger.perf_metrics = pd.read_csv(file_path)

    @staticmethod
    def perfmon_step():
        Logger.save_perf_metrics(folder_name="backups")
        Logger.perf_rep_counter = 0

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
                    msg_string += f" time(): {execution_time:.2f} s"
            print(msg_string, "::", *args)
            return start_time
        
    def log_perf(*args, mode=1, repetition_ID=None, test_suite_ID=None, variation=None, _input=None, prev_start_time=None, prev_perf=None, prev_proc=None):
        if Logger.measure_perf:
            start_time = start_perf_time = start_process_time = None
            msg_string = f"{cyan}[{Logger.perf_rep_counter}]{blk} Perf @ {Logger.mode2flag[mode]}"
            if mode==1:
                start_time = time()
                start_perf_time = perf_counter()
                start_process_time = process_time()
            elif mode==2 and prev_start_time is not None:
                end_time = time()
                end_perf_time = perf_counter()
                end_process_time = process_time()
                execution_time = end_time - prev_start_time
                perf_delta = end_perf_time - prev_perf
                process_time_delta = end_process_time - prev_proc
                msg_string += f" [time: {execution_time:.9f} s]" + f" [perf_counter: {perf_delta:.9f} s]" + f" [process_time: {process_time_delta:.9f} s]"
                Logger.add_to_perf_metrics(*args, test_suite_ID=test_suite_ID, repetition_ID=repetition_ID, variation=variation, _input=_input, _time=execution_time, _perf_counter=perf_delta, _process_time=process_time_delta)
        print(msg_string, "::", *args)
        return start_time, start_perf_time, start_process_time

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
    for param in model.parameters():
        param.grad = None
    return model.to(device)