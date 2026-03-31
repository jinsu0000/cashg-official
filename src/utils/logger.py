import time
import os
import sys
import numpy as np
import torch

_print_once_seen = set()
TRACE_LOG = False

def set_trace_log(enable=True):
    global TRACE_LOG
    TRACE_LOG = enable
    print(f"[TRACE_LOG] Set to {enable}")

def is_trace_log():
    global TRACE_LOG
    return TRACE_LOG

def print_once(*args, **kwargs):
    msg = " ".join(map(str, args))
    if msg not in _print_once_seen:
        _print_once_seen.add(msg)
        print(*args, **kwargs)
        sys.stdout.flush()

def print_trace(*args, **kwargs):
    global TRACE_LOG
    if TRACE_LOG:
        print("[TRACE]", *args, **kwargs)
        sys.stdout.flush()

def log_stats(name, tensor):

    mean_val = tensor.mean().item()
    var_val = tensor.var().item()
    print(f"{name} → mean: {mean_val:.4f}, var: {var_val:.4f}")

def debug_coord_stats(tag, arr):
    arr = arr.cpu().numpy() if torch.is_tensor(arr) else np.array(arr)
    if arr.shape[0] == 0:
        print(f"[{tag}] shape={arr.shape}, EMPTY")
        return
    print(f"[{tag}] shape={arr.shape}, min={arr[:, :2].min(axis=0)}, max={arr[:, :2].max(axis=0)}, mean={arr[:, :2].mean(axis=0)}, std={arr[:, :2].std(axis=0)}")

def trace_if_abnormal(name, tensor, nan_thr=1e-6, inf_thr=1e6, step=None):
    global TRACE_LOG
    is_nan = torch.isnan(tensor).any().item()
    is_inf = torch.isinf(tensor).any().item()
    is_large = tensor.abs().max().item() > inf_thr
    if is_nan or is_inf or is_large:
        if not TRACE_LOG:
            print(f"[{name}] !!!!TRACE START!!!!")
            set_trace_log(True)
        print_trace(f"[{name}] step={step} NAN/INF/LARGE detected.")
        print_trace(f"    min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")
        print_trace(f"    slice: {tensor.flatten()[:10].detach().cpu().numpy()}")
        return True
    return False

def check_tensor_valid(name, tensor, step=None, nan_thr=1e-6, inf_thr=1e6):
    if not torch.is_tensor(tensor):
        print_trace(f"[{name}] is not a tensor (type={type(tensor)})")
        return False
    is_nan = torch.isnan(tensor).any().item()
    is_inf = torch.isinf(tensor).any().item()
    is_large = tensor.abs().max().item() > inf_thr
    if is_nan or is_inf or is_large:
        if not TRACE_LOG:
            print(f"[{name}] !!!!TRACE START!!!!")
            set_trace_log(True)
        print_trace(f"[{name}] (step={step}) anomaly detected (nan={is_nan}, inf={is_inf}, large={is_large})")
        print_trace(f"  min={tensor.min().item():.4e}, max={tensor.max().item():.4e}, mean={tensor.mean().item():.4e}, std={tensor.std().item():.4e}")
        print_trace(f"  slice: {tensor.flatten()[:10].detach().cpu().numpy()}")
        return True
    return False

def check_data_valid(name, data, step=None):
    try:
        arr = data.detach().cpu().numpy() if torch.is_tensor(data) else np.array(data)
    except Exception as e:
        if not TRACE_LOG:
            print(f"[{name}] !!!!TRACE START!!!!")
            set_trace_log(True)
        print_trace(f"[{name}] step={step} tensor detach error: {e}")
        return False

    is_nan = np.isnan(arr).any()
    is_inf = np.isinf(arr).any()
    is_large = np.abs(arr).max() > 1e6
    if is_nan or is_inf or is_large:
        if not TRACE_LOG:
            print(f"[{name}] !!!!TRACE START!!!!")
            set_trace_log(True)
        print_trace(f"[{name}] (step={step}) data anomaly (nan={is_nan}, inf={is_inf}, large={is_large})")
        print_trace(f"  min={arr.min():.4e}, max={arr.max():.4e}, mean={arr.mean():.4e}, std={arr.std():.4e}")
        print_trace(f"  slice: {arr.flatten()[:10]}")
        return True
    return False

def check_explosion(name, tensor, step=None, thres=1e3, warmup=500):
    if step is not None and step < warmup:
        return False
    if not torch.is_tensor(tensor):
        return False
    val = tensor.detach().cpu().float().abs().max().item()
    if val > thres:
        if not TRACE_LOG:
            print(f"[{name}] !!!!TRACE START!!!!")
            set_trace_log(True)
        print_trace(f"[{name}] (step={step}) EXPLOSION: value={val:.4e} (thres={thres:.1e})")
        return True
    return False

def print_traj_stats(tag, traj):
    arr = traj
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    print(f"[{tag}] shape={arr.shape}")
    print(f"  x min/max/mean/std: {arr[:,0].min():.4f} ~ {arr[:,0].max():.4f}, {arr[:,0].mean():.4f}, {arr[:,0].std():.4f}")
    print(f"  y min/max/mean/std: {arr[:,1].min():.4f} ~ {arr[:,1].max():.4f}, {arr[:,1].mean():.4f}, {arr[:,1].std():.4f}")
    if arr.shape[1] > 2:
        print(f"  extra min/max/mean/std: {arr[:,2].min():.4f} ~ {arr[:,2].max():.4f}, {arr[:,2].mean():.4f}, {arr[:,2].std():.4f}")
