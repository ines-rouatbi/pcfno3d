"""Short synthetic CPU capacity check; does not train or modify a checkpoint."""
import copy
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from pcfno3d.config import Config
from pcfno3d.model import build_model, count_parameters


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (name, ctypes.c_size_t) for name in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


name = sys.argv[1] if len(sys.argv) > 1 else 'tunisie_fno3d'
fine = name.endswith('_land')
shape, batch = ((25, 78, 45), 1) if fine else ((22, 32, 19), 4)
torch.set_num_threads(4)
torch.manual_seed(42)
cfg = Config.load(name)
model = build_model(cfg, 25)
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
x = torch.randn(batch, 25, *shape)
target = torch.randn(batch, 3, *shape)
elapsed = []
for step in range(4):
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for p, q in zip(model.parameters(), ema.parameters()):
            q.lerp_(p, .001)
    duration = time.perf_counter() - start
    if step:
        elapsed.append(duration)
    print(json.dumps({'step': step, 'seconds': duration}), flush=True)
mem = ProcessMemory()
mem.cb = ctypes.sizeof(mem)
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(mem), mem.cb)
result = {
    'configuration': name, 'device': 'cpu', 'threads': 4, 'batch': batch,
    'shape': shape, 'parameters_real_equivalent': count_parameters(model),
    'weight_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
    'median_step_seconds': statistics.median(elapsed),
    'peak_working_set_gib': mem.PeakWorkingSetSize / 2**30 if ok else None,
    'peak_commit_gib': mem.PeakPagefileUsage / 2**30 if ok else None,
    'limitations': 'Synthetic model + MSE + AdamW + EMA. Excludes real data, physics losses, validation and checkpoint I/O. CPU memory is not GPU VRAM.',
}
Path(f'output/redesign/capacity-{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
