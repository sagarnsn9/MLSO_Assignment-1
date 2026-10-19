"""
ML System Optimization (AIMLZG516) - Group Assignment
=====================================================
Synchronous data-parallel training of a CIFAR-10 ResNet with PyTorch DistributedDataParallel,
instrumented to measure speedup, parallel efficiency, communication cost, response time and
accuracy gap.  Everything needed for the assignment lives in this ONE file:

    model (ResNet-10, CIFAR variant)  ->  data pipeline  ->  DDP worker  ->  experiment driver
    ->  correctness verification  ->  summary tables (CSV)  ->  plots (PNG)

Design in one paragraph
-----------------------
Every worker is one OS process that holds a full replica of the model (data parallelism).  A
DistributedSampler gives each worker a disjoint 1/N shard of every epoch, the GLOBAL batch is kept
fixed and split equally (strong scaling, so the optimisation maths is identical for every N), and
the gradients are averaged with a ring all-reduce once per step (synchronous SGD).  Three
communication modes are implemented:

    manual    - backward runs under ``no_sync()``; then a barrier (measures load imbalance) and ONE
                flat all-reduce (measures pure communication) are timed separately from compute.
    ddp       - stock DDP: 25 MB gradient buckets all-reduced *during* backward (overlap).
    ddp_fp16  - stock DDP + fp16 gradient-compression communication hook (halves bytes on the wire).

CPU workers use the gloo backend with a FIXED number of intra-op threads per worker (no thread
over-subscription); GPU workers use NCCL where available (Linux) and gloo elsewhere.

Usage
-----
    python mlso_cifar10_ddp.py smoke        # ~1 min end-to-end check on random data (no download)
    python mlso_cifar10_ddp.py verify       # correctness tests (gradient equivalence, replica sync)
    python mlso_cifar10_ddp.py commbench    # isolated all-reduce micro-benchmark -> alpha/beta per N
    python mlso_cifar10_ddp.py scaling      # 1/2/4/8 workers, manual comm  -> S(N), E(N), comm %
    python mlso_cifar10_ddp.py comm         # 2/4/8 workers with ddp and ddp_fp16 comm modes
    python mlso_cifar10_ddp.py batch        # global batch 1024 (vs 256 in 'scaling'), 1 and 8 workers
    python mlso_cifar10_ddp.py affinity     # 'scaling' with each worker pinned to its own cores
    python mlso_cifar10_ddp.py final        # full CIFAR-10, 8 CPU workers, 30 epochs
    python mlso_cifar10_ddp.py final_gpu    # full CIFAR-10, 1 GPU, same recipe (platform comparison)
    python mlso_cifar10_ddp.py plot         # all figures + results/summary_all.csv from results/*.json
    python mlso_cifar10_ddp.py all          # everything above, in order

Any preset value can be overridden, e.g. ``scaling --workers 1 2 4 --epochs 2 --threads 1``.
Run ``python mlso_cifar10_ddp.py -h`` for the full option list.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

LOG = logging.getLogger("mlso")

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


# =============================================================================================
# 1. Model - ResNet adapted to 32x32 inputs
# =============================================================================================
class BasicBlock(nn.Module):
    """Two 3x3 conv-BN layers with an identity (or 1x1 projection) shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem with stride 1 and no max-pool (the ImageNet 7x7 stem + pool would
    shrink a 32x32 image to 8x8 before the first residual stage), widths 32-64-128-256."""

    def __init__(self, block: type, num_blocks: Sequence[int], num_classes: int = 10, width: int = 32):
        super().__init__()
        self.in_channels = width
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, width * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, width * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, width * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8 * block.expansion, num_classes)

    def _make_layer(self, block: type, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def ResNet10(num_classes: int = 10) -> ResNet:
    """One BasicBlock per stage: ~1.23 M parameters (4.7 MB of fp32 gradients per step)."""
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)


def ResNet18(num_classes: int = 10) -> ResNet:
    """Standard-depth variant (kept for completeness; ~2.8 M params at width 32)."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


MODELS = {"resnet10": ResNet10, "resnet18": ResNet18}


# =============================================================================================
# 2. Experiment presets and command-line interface
# =============================================================================================
COMMON_DEFAULTS: Dict[str, Any] = {
    "model": "resnet10",
    "workers": [1],
    "epochs": 3,
    "batch_size": 256,       # GLOBAL batch, split equally between workers
    "lr": None,              # None -> 0.2 * batch_size / 256  (linear scaling rule)
    "train_size": 20000,     # 0 = all 50,000 images
    "test_size": 0,          # 0 = all 10,000 images
    "threads": 0,            # intra-op threads per CPU worker, 0 = physical cores / max(workers)
    "comm": "manual",
    "device": "auto",
    "loader": "auto",        # torchvision (PIL, CPU) | tensor (whole dataset as a tensor on device)
    "loader_workers": 0,
    "amp": False,
    "pin_cores": False,
    "fake_data": False,
    "augment": True,
    "save": True,
}

# Each command is a list of runs; a run = one tag, executed once per worker count.
EXPERIMENTS: Dict[str, List[Dict[str, Any]]] = {
    "smoke": [dict(tag="smoke", workers=[1, 2], epochs=1, train_size=512, test_size=256,
                   batch_size=64, fake_data=True, save=False)],
    "scaling": [dict(tag="scaling", workers=[1, 2, 4, 8], epochs=5, train_size=0)],
    "comm": [dict(tag="comm_ddp", workers=[2, 4, 8], epochs=5, train_size=0, comm="ddp"),
             dict(tag="comm_ddp_fp16", workers=[2, 4, 8], epochs=5, train_size=0, comm="ddp_fp16")],
    "batch": [dict(tag="batch_b1024", workers=[1, 8], epochs=5, train_size=0, batch_size=1024)],
    "affinity": [dict(tag="scaling_pinned", workers=[1, 2, 4, 8], epochs=3, train_size=0, pin_cores=True)],
    "final": [dict(tag="final_cpu", workers=[8], epochs=30, train_size=0)],
    "final_gpu": [dict(tag="final_gpu", workers=[1], epochs=30, train_size=0, device="cuda")],
}
PIPELINE = ["smoke", "verify", "commbench", "scaling", "comm", "batch", "affinity", "final", "final_gpu", "plot"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CIFAR-10 data-parallel training with PyTorch DDP (MLSO assignment)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command", choices=list(EXPERIMENTS) + ["verify", "commbench", "plot", "all"])
    g = p.add_argument_group("overrides (default: value from the preset)")
    g.add_argument("--workers", type=int, nargs="+", help="worker counts to run, e.g. 1 2 4 8")
    g.add_argument("--epochs", type=int)
    g.add_argument("--batch_size", type=int, help="GLOBAL batch size")
    g.add_argument("--lr", type=float, help="peak OneCycle LR (default 0.2 per 256 images)")
    g.add_argument("--train_size", type=int, help="0 = full 50,000 images")
    g.add_argument("--test_size", type=int, help="0 = full 10,000 images")
    g.add_argument("--threads", type=int, help="intra-op threads per CPU worker (0 = auto)")
    g.add_argument("--comm", choices=["manual", "ddp", "ddp_fp16"])
    g.add_argument("--device", choices=["auto", "cpu", "cuda"])
    g.add_argument("--loader", choices=["auto", "torchvision", "tensor"])
    g.add_argument("--loader_workers", type=int, help="DataLoader processes per worker (torchvision loader)")
    g.add_argument("--model", choices=list(MODELS))
    g.add_argument("--amp", action="store_true", default=None, help="bf16 autocast (CUDA only)")
    g.add_argument("--pin_cores", action="store_true", default=None,
                   help="pin worker r to its own block of CPU cores (fastest cores first)")
    g.add_argument("--fake_data", action="store_true", default=None)
    g.add_argument("--no_augment", action="store_true")
    g.add_argument("--no_save", action="store_true")
    g.add_argument("--tag", help="name of the result files (single-run commands only)")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify_workers", type=int, nargs="+", default=[2, 4])
    p.add_argument("--cooldown", type=int, default=0,
                   help="idle seconds before every run (limits thermal carry-over between configurations)")
    return p


def resolve_run(cli: argparse.Namespace, preset: Dict[str, Any]) -> argparse.Namespace:
    """Merge COMMON_DEFAULTS <- preset <- command-line overrides into one namespace."""
    cfg = dict(COMMON_DEFAULTS)
    cfg.update(preset)
    for key in list(cfg):
        value = getattr(cli, key, None)
        if value is not None and key != "tag":
            cfg[key] = value
    if cli.tag:
        cfg["tag"] = cli.tag
    if cli.no_augment:
        cfg["augment"] = False
    if cli.no_save:
        cfg["save"] = False
    if cfg["lr"] is None:
        cfg["lr"] = 0.2 * cfg["batch_size"] / 256
    cfg.update(data_dir=cli.data_dir, out_dir=cli.out_dir, seed=cli.seed, cooldown=cli.cooldown)
    return argparse.Namespace(**cfg)


# =============================================================================================
# 3. Environment helpers (devices, threads, metadata)
# =============================================================================================
def setup_logging(rank: Optional[int] = None) -> None:
    prefix = "" if rank is None else f"[rank {rank}] "
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {prefix}%(message)s",
                        datefmt="%H:%M:%S", force=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def cpu_name() -> str:
    """Human-readable CPU model name on Windows / Linux / macOS."""
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # metadata only - never fail a run because of it
        pass
    return platform.processor() or "unknown"


def cpu_core_classes() -> Optional[Dict[int, int]]:
    """Windows only: logical CPU -> EfficiencyClass (higher = faster core, e.g. Intel P-core = 1,
    E-core = 0).  Used to pin workers to the fastest cores first and to report the topology."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        n = wintypes.ULONG(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
        buf = (ctypes.c_ubyte * n.value)()
        if not k32.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0):
            return None
        classes, off = {}, 0
        while off < n.value:   # SYSTEM_CPU_SET_INFORMATION records: LogicalProcessorIndex @14, EfficiencyClass @18
            classes[buf[off + 14]] = buf[off + 18]
            off += int.from_bytes(bytes(buf[off:off + 4]), "little")
        return classes
    except Exception:
        return None


def disable_os_power_throttling() -> bool:
    """Windows 11 'EcoQoS': processes started from a background console may be classified as
    low-QoS and parked on E-cores at reduced clock.  Opt this process out (execution-speed and
    timer-resolution throttling) and raise its priority, so timings reflect the hardware rather
    than the OS power policy.  Returns True if the opt-out succeeded."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        class ProcessPowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

        state = ProcessPowerThrottlingState(1, 0x1 | 0x4, 0)   # control EXECUTION_SPEED | IGNORE_TIMER_RES, state off
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        ok = bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4,  # 4 = ProcessPowerThrottling
                                            ctypes.byref(state), ctypes.sizeof(state)))
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        return ok
    except Exception:
        return False


def cores_fastest_first() -> List[int]:
    classes = cpu_core_classes()
    cores = list(range(psutil.cpu_count(logical=True) or 1))
    return sorted(cores, key=lambda c: (-classes.get(c, 0), c)) if classes else cores


def machine_info() -> Dict[str, Any]:
    info = {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gloo": dist.is_gloo_available(),
        "nccl": dist.is_nccl_available(),
    }
    classes = cpu_core_classes()
    if classes:
        info["fast_cores"] = sorted(c for c, e in classes.items() if e == max(classes.values()))
        info["efficient_cores"] = sorted(c for c, e in classes.items() if e != max(classes.values()))
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def pick_device_and_workers(args: argparse.Namespace) -> List[int]:
    """Resolve device / loader / threads and drop worker counts this machine cannot run."""
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.loader == "auto":
        args.loader = "tensor" if args.device == "cuda" else "torchvision"
    if args.amp and args.device != "cuda":
        LOG.info("--amp is only used on CUDA; ignoring it on CPU.")
        args.amp = False

    if args.device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("--device cuda was requested but no CUDA GPU is visible to PyTorch.")
        max_workers = torch.cuda.device_count()
        workers = [w for w in args.workers if w <= max_workers]
    else:
        logical = os.cpu_count() or 1
        physical = psutil.cpu_count(logical=False) or logical
        if args.threads == 0:
            # the largest configuration should use exactly the physical cores, never more
            usable = [w for w in args.workers if w <= physical] or [physical]
            args.threads = max(1, physical // max(usable))
        max_workers = max(1, physical // args.threads)
        workers = [w for w in args.workers if w <= max_workers]

    skipped = sorted(set(args.workers) - set(workers))
    if skipped:
        LOG.info("Skipping worker counts %s: not enough %s.", skipped,
                 "GPUs" if args.device == "cuda" else "physical cores")
    if not workers:
        workers = [max_workers]
    for w in workers:
        if args.batch_size % w:
            raise ValueError(f"Global batch {args.batch_size} is not divisible by {w} workers.")
    return workers


def find_free_port() -> int:
    """Ask the OS for an unused TCP port so back-to-back runs never collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init_distributed(rank: int, world_size: int, port: int, device: torch.device) -> str:
    """Join the process group.  A 1-worker run stays a plain process (no DDP wrapper, no
    collectives), so T1 is a true sequential baseline that carries no distributed overhead."""
    if world_size == 1:
        return "none"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return backend


def configure_worker(rank: int, world_size: int, args: argparse.Namespace) -> torch.device:
    """Device selection and CPU thread / affinity control for one worker process."""
    args.os_throttling_disabled = disable_os_power_throttling()
    if args.device == "cuda":
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return device

    # KEY to CPU scaling: each worker gets a fixed thread budget, so N workers use N*T threads
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # can only be set once per process
        pass
    if args.pin_cores:
        cores = cores_fastest_first()[rank * args.threads:(rank + 1) * args.threads]
        try:
            psutil.Process().cpu_affinity(cores)
        except (AttributeError, psutil.Error, ValueError) as exc:
            LOG.warning("CPU pinning not supported here (%s); continuing unpinned.", exc)
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """CUDA kernels are asynchronous: wait for them before reading a wall clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# =============================================================================================
# 4. Data pipeline
# =============================================================================================
def _cifar(args: argparse.Namespace, train: bool, transform=None):
    return datasets.CIFAR10(root=args.data_dir, train=train, download=False, transform=transform)


def ensure_cifar10(data_dir: str) -> None:
    """Download once in the parent process so workers never race on the download."""
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)


def raw_arrays(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """uint8 NCHW image tensors + int64 labels for train and test (honours train/test_size)."""
    if args.fake_data:
        g = torch.Generator().manual_seed(0)
        n_tr, n_te = args.train_size or 512, args.test_size or 256
        return (torch.randint(0, 256, (n_tr, 3, 32, 32), dtype=torch.uint8, generator=g),
                torch.randint(0, 10, (n_tr,), generator=g),
                torch.randint(0, 256, (n_te, 3, 32, 32), dtype=torch.uint8, generator=g),
                torch.randint(0, 10, (n_te,), generator=g))
    out = []
    for train, size in ((True, args.train_size), (False, args.test_size)):
        ds = _cifar(args, train)
        n = size if size > 0 else len(ds)
        out.append(torch.from_numpy(ds.data[:n]).permute(0, 3, 1, 2).contiguous())
        out.append(torch.tensor(ds.targets[:n], dtype=torch.long))
    return tuple(out)  # type: ignore[return-value]


class DeviceTensorLoader:
    """Keeps the whole uint8 dataset on the target device and does crop/flip/normalise there.

    CIFAR-10 is only 150 MB as uint8, so on a GPU this removes the PIL/CPU data-loading bottleneck
    (and the Windows multiprocessing DataLoader start-up cost) entirely.  Index order still comes
    from a (Distributed)Sampler, so sharding semantics are identical to the torchvision path.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int, sampler,
                 device: torch.device, augment: bool, drop_last: bool, channels_last: bool = False):
        self.x = images.to(device)
        self.y = labels.to(device)
        self.batch_size, self.sampler, self.device = batch_size, sampler, device
        self.augment, self.drop_last, self.channels_last = augment, drop_last, channels_last
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
        self.ar = torch.arange(32, device=device)
        self.gen = torch.Generator(device=device)

    def set_seed(self, seed: int) -> None:
        self.gen.manual_seed(seed)

    def __len__(self) -> int:
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """RandomCrop(32, padding=4) + RandomHorizontalFlip, vectorised over the batch."""
        b = x.size(0)
        x = F.pad(x, (4, 4, 4, 4))  # zero padding, same as torchvision on the [0, 1] image
        oy = torch.randint(0, 9, (b, 1), device=self.device, generator=self.gen)
        ox = torch.randint(0, 9, (b, 1), device=self.device, generator=self.gen)
        rows, cols = (oy + self.ar)[:, None, :, None], (ox + self.ar)[:, None, None, :]
        x = x[torch.arange(b, device=self.device)[:, None, None, None],
              torch.arange(3, device=self.device)[None, :, None, None], rows, cols]
        flip = torch.rand(b, device=self.device, generator=self.gen) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        idx = torch.as_tensor(list(iter(self.sampler)), device=self.device)
        for b in range(len(self)):
            chunk = idx[b * self.batch_size:(b + 1) * self.batch_size]
            x = self.x[chunk].float().div_(255)
            if self.augment:
                x = self._augment(x)
            x = (x - self.mean) / self.std
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            yield x, self.y[chunk]


class IndexList(list):
    """A plain list of indices usable as a sampler (has __len__ and __iter__)."""


def build_loaders(args: argparse.Namespace, rank: int, world_size: int, device: torch.device):
    """Return (train_loader, test_loader, train_sampler, n_train, n_test) for one worker.

    Training: DistributedSampler -> disjoint shard of size n/N per worker, drop_last so that every
    worker runs exactly the same number of steps (otherwise the all-reduce would deadlock).
    Testing:  worker r evaluates images r, r+N, r+2N, ... (no duplicates, no padding), then the
    correct counts are summed with an all-reduce, so accuracy is over the full test set.
    """
    local_batch = args.batch_size // world_size
    if args.loader == "tensor":
        xtr, ytr, xte, yte = raw_arrays(args)
        n_train, n_test = len(ytr), len(yte)
        sampler = DistributedSampler(range(n_train), num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=args.seed, drop_last=True)
        cl = device.type == "cuda"
        train_loader = DeviceTensorLoader(xtr, ytr, local_batch, sampler, device, args.augment, True, cl)
        train_loader.set_seed(args.seed + rank)
        test_loader = DeviceTensorLoader(xte, yte, 512, IndexList(range(rank, n_test, world_size)),
                                         device, False, False, cl)
        return train_loader, test_loader, sampler, n_train, n_test

    normalize = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if args.augment else []
    train_tf, test_tf = transforms.Compose(aug + normalize), transforms.Compose(normalize)
    if args.fake_data:
        train_ds = datasets.FakeData(size=args.train_size or 512, image_size=(3, 32, 32), num_classes=10,
                                     transform=train_tf)
        test_ds = datasets.FakeData(size=args.test_size or 256, image_size=(3, 32, 32), num_classes=10,
                                    transform=test_tf, random_offset=1000)
    else:
        train_ds, test_ds = _cifar(args, True, train_tf), _cifar(args, False, test_tf)
        if args.train_size > 0:
            train_ds = Subset(train_ds, range(args.train_size))
        if args.test_size > 0:
            test_ds = Subset(test_ds, range(args.test_size))

    pin = device.type == "cuda"
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=local_batch, sampler=sampler, drop_last=True,
                              num_workers=args.loader_workers, pin_memory=pin,
                              persistent_workers=args.loader_workers > 0)
    test_part = Subset(test_ds, list(range(rank, len(test_ds), world_size)))
    test_loader = DataLoader(test_part, batch_size=512, shuffle=False,
                             num_workers=args.loader_workers, pin_memory=pin)
    return train_loader, test_loader, sampler, len(train_ds), len(test_ds)


# =============================================================================================
# 5. Distributed training primitives
# =============================================================================================
def allreduce_gradients(params: Sequence[torch.Tensor], world_size: int) -> None:
    """Average gradients across workers with ONE all-reduce on a flat buffer.

    One large message instead of ~60 small ones amortises per-message latency (alpha term of the
    alpha-beta cost model); the ring algorithm then moves 2(N-1)/N * |g| bytes per worker.
    """
    grads = [p.grad for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset:offset + n].view_as(g))
        offset += n


class LocalReplica(nn.Module):
    """Single-process stand-in for DDP exposing the same ``.module`` / ``no_sync()`` interface."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *a, **kw):
        return self.module(*a, **kw)

    def no_sync(self):
        return nullcontext()


def wrap_model(model: nn.Module, device: torch.device, comm: str) -> nn.Module:
    if not dist.is_initialized():
        return LocalReplica(model)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    if comm == "ddp_fp16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        ddp.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)
    return ddp


def barrier(device: torch.device) -> None:
    if not dist.is_initialized():
        return
    if device.type == "cuda" and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def average_bn_buffers(module: nn.Module, world_size: int) -> None:
    """Make BatchNorm running statistics identical on all replicas before evaluation.

    Gradients are synchronised but BN running mean/var are per-worker (each sees only its shard);
    averaging them gives the estimate a single worker on the full batch would have produced.
    """
    if not dist.is_initialized():
        return
    for buf in module.buffers():
        if buf.is_floating_point():
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf.div_(world_size)
        else:
            dist.broadcast(buf, src=0)


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer: optim.Optimizer, scheduler,
                    device: torch.device, args: argparse.Namespace, world_size: int,
                    params: Sequence[torch.Tensor]) -> Dict[str, Any]:
    """One epoch with a per-step time breakdown: data | compute | wait (imbalance) | comm."""
    model.train()
    timers = {"data": 0.0, "compute": 0.0, "wait": 0.0, "comm": 0.0}
    step_times: List[float] = []
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total, steps = 0, 0
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.amp else nullcontext())

    epoch_start = t_last = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sync_device(device)
        t1 = time.perf_counter()
        timers["data"] += t1 - t_last
        optimizer.zero_grad(set_to_none=True)

        if args.comm == "manual":
            with model.no_sync():                       # local forward/backward, no gradient sync
                with autocast:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                loss.backward()
            sync_device(device)
            t2 = time.perf_counter()
            timers["compute"] += t2 - t1
            if world_size > 1:
                barrier(device)                         # sync point 1a: wait for the slowest worker
                t3 = time.perf_counter()
                timers["wait"] += t3 - t2
                allreduce_gradients(params, world_size)  # sync point 1b: gradient averaging
                sync_device(device)
                t4 = time.perf_counter()
                timers["comm"] += t4 - t3
            else:
                t4 = t2
        else:                                           # DDP buckets overlap all-reduce with backward
            with autocast:
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            sync_device(device)
            t4 = time.perf_counter()
            timers["compute"] += t4 - t1

        optimizer.step()                                # identical averaged gradient on all ranks
        scheduler.step()
        loss_sum += loss.detach().double() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum()
        total += labels.size(0)
        steps += 1
        sync_device(device)
        t_now = time.perf_counter()
        timers["compute"] += t_now - t4
        step_times.append(t_now - t_last)
        t_last = t_now

    return {"loss_sum": loss_sum.item(), "correct": correct.item(), "total": total, "steps": steps,
            "epoch_time": time.perf_counter() - epoch_start, "timers": timers, "step_times": step_times}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, channels_last: bool = False) -> Tuple[float, float]:
    """Each worker tests its own disjoint slice; loss/correct/total are summed over workers."""
    model.eval()
    stats = torch.zeros(3, dtype=torch.float64, device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        outputs = model(images).float()
        stats[0] += F.cross_entropy(outputs, labels, reduction="sum").double()
        stats[1] += (outputs.argmax(1) == labels).sum()
        stats[2] += labels.size(0)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats[0].item() / stats[2].item(), 100.0 * stats[1].item() / stats[2].item()


def all_reduce_values(values: Sequence[float], device: torch.device, op=dist.ReduceOp.SUM) -> List[float]:
    t = torch.tensor(list(values), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=op)
    return t.tolist()


# =============================================================================================
# 6. Worker process: one rank of a training run
# =============================================================================================
def main_worker(rank: int, world_size: int, args: argparse.Namespace, port: int, result_file: str) -> None:
    setup_logging(rank if rank else None)
    device = configure_worker(rank, world_size, args)
    backend = init_distributed(rank, world_size, port, device)
    seed_everything(args.seed)

    train_loader, test_loader, train_sampler, n_train, n_test = build_loaders(args, rank, world_size, device)
    local_batch = args.batch_size // world_size
    channels_last = device.type == "cuda"

    torch.manual_seed(args.seed)                     # identical initial weights (DDP also broadcasts)
    net = MODELS[args.model](num_classes=10).to(device)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    model = wrap_model(net, device, args.comm)
    seed_everything(args.seed + rank)                # different augmentation stream per worker

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    grad_mb = n_params * 4 / 1024 ** 2
    wire_mb = grad_mb / 2 if args.comm == "ddp_fp16" else grad_mb
    ring_mb_per_step = 2 * (world_size - 1) / world_size * wire_mb   # bytes each worker sends

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                              steps_per_epoch=len(train_loader), pct_start=0.25,
                                              cycle_momentum=False)
    if rank == 0:
        LOG.info("%s | %d worker(s) on %s (%s) | train %d / test %d images | global batch %d (%d/worker) "
                 "| %d steps/epoch | %s params (%.2f MB grads) | comm=%s",
                 args.tag, world_size, device.type, backend, n_train, n_test, local_batch * world_size,
                 local_batch, len(train_loader), f"{n_params:,}", grad_mb, args.comm)

    proc = psutil.Process(os.getpid())
    history: List[Dict[str, Any]] = []
    step_times_after_warmup: List[float] = []
    best_acc, train_time_so_far, peak_rss = 0.0, 0.0, 0.0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        ep = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args, world_size, params)
        if epoch > 0 or args.epochs == 1:
            step_times_after_warmup.extend(ep["step_times"])

        average_bn_buffers(model.module, world_size)  # sync point 2 (once per epoch)
        t0 = time.perf_counter()
        test_loss, test_acc = evaluate(model.module, test_loader, device, channels_last)
        eval_time = time.perf_counter() - t0

        tm = ep["timers"]
        loss_sum, correct, total = all_reduce_values([ep["loss_sum"], ep["correct"], ep["total"]], device)
        epoch_time, eval_time, compute_max = all_reduce_values(
            [ep["epoch_time"], eval_time, tm["compute"]], device, dist.ReduceOp.MAX)
        data_t, compute_t, wait_t, comm_t = [v / world_size for v in all_reduce_values(
            [tm["data"], tm["compute"], tm["wait"], tm["comm"]], device)]
        peak_rss = max(peak_rss, all_reduce_values([proc.memory_info().rss / 1024 ** 2], device)[0])
        train_time_so_far += epoch_time               # an epoch ends when the slowest worker ends

        row = {"epoch": epoch + 1, "train_loss": loss_sum / total, "train_acc": 100.0 * correct / total,
               "test_loss": test_loss, "test_acc": test_acc, "epoch_time": epoch_time, "eval_time": eval_time,
               "data_time": data_t, "compute_time": compute_t, "compute_time_max": compute_max,
               "wait_time": wait_t, "comm_time": comm_t, "train_time_so_far": train_time_so_far,
               "steps": ep["steps"], "lr_end": scheduler.get_last_lr()[0]}
        history.append(row)

        if rank == 0:
            LOG.info("epoch %2d/%d | train loss %.4f acc %.2f%% | test loss %.4f acc %.2f%% | %.1fs "
                     "(data %.1f, compute %.1f, wait %.1f, comm %.1f)", epoch + 1, args.epochs,
                     row["train_loss"], row["train_acc"], test_loss, test_acc, epoch_time,
                     data_t, compute_t, wait_t, comm_t)
            if test_acc > best_acc:
                best_acc = test_acc
                if args.save:
                    torch.save(model.module.state_dict(),
                               os.path.join(args.out_dir, f"best_{args.tag}_w{world_size}.pth"))

    if rank == 0:
        later = history[1:] if len(history) > 1 else history     # exclude warm-up epoch
        steps_total = sum(h["steps"] for h in history)
        tot = lambda k: sum(h[k] for h in history)               # noqa: E731
        comm_per_step = tot("comm_time") / max(steps_total, 1)
        result = {
            "tag": args.tag, "workers": world_size, "device": device.type, "backend": backend,
            "model": args.model, "threads_per_worker": args.threads if device.type == "cpu" else None,
            "pin_cores": bool(args.pin_cores), "os_throttling_disabled": args.os_throttling_disabled,
            "epoch_time_cv": (statistics.pstdev([h["epoch_time"] for h in later]) /
                              statistics.mean([h["epoch_time"] for h in later])),
            "comm_mode": args.comm, "loader": args.loader, "amp": bool(args.amp),
            "global_batch": local_batch * world_size, "local_batch": local_batch, "lr": args.lr,
            "epochs": args.epochs, "train_images": n_train, "test_images": n_test, "augment": args.augment,
            "params": n_params, "grad_size_mb": grad_mb, "ring_mb_per_step": ring_mb_per_step,
            "total_train_time": tot("epoch_time"),
            "avg_epoch_time": tot("epoch_time") / len(history),
            "avg_epoch_time_no_warmup": sum(h["epoch_time"] for h in later) / len(later),
            "comm_fraction": tot("comm_time") / tot("epoch_time"),
            "wait_fraction": tot("wait_time") / tot("epoch_time"),
            "data_fraction": tot("data_time") / tot("epoch_time"),
            "compute_fraction": tot("compute_time") / tot("epoch_time"),
            "compute_imbalance": tot("compute_time_max") / max(tot("compute_time"), 1e-12),
            "comm_per_step_ms": 1000.0 * comm_per_step,
            "bus_bandwidth_gbs": (ring_mb_per_step / 1024) / comm_per_step if comm_per_step > 0 else None,
            "step_time_p50_ms": 1000.0 * statistics.median(step_times_after_warmup),
            "step_time_p95_ms": 1000.0 * float(np.percentile(step_times_after_warmup, 95)),
            "final_test_acc": history[-1]["test_acc"], "best_test_acc": best_acc,
            "peak_rss_mb_all_workers": peak_rss,
            "machine": machine_info(), "history": history,
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        LOG.info("best test accuracy %.2f%%", best_acc)

    if dist.is_initialized():
        barrier(device)
        dist.destroy_process_group()


# =============================================================================================
# 7. Correctness verification
# =============================================================================================
def verify_worker(rank: int, world_size: int, args: argparse.Namespace, port: int, result_file: str) -> None:
    """(a) gradient equivalence of every comm mode vs single-process full-batch gradient,
    (b) replica consistency of parameters after real training steps with augmentation."""
    setup_logging(rank if rank else None)
    device = configure_worker(rank, world_size, args)
    init_distributed(rank, world_size, port, device)
    xtr, ytr, _, _ = raw_arrays(args)
    mean, std = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1), torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    B = 64
    x_all = ((xtr[:B].float() / 255 - mean) / std).to(device)
    y_all = ytr[:B].to(device)
    x_loc, y_loc = x_all[rank::world_size], y_all[rank::world_size]   # this worker's shard of the batch
    report: Dict[str, Any] = {"world_size": world_size, "global_batch": B, "gradient_equivalence": {}}

    def fresh_model() -> nn.Module:
        torch.manual_seed(args.seed)
        return MODELS[args.model](num_classes=10).to(device)

    def flat_grad(m: nn.Module) -> torch.Tensor:
        return torch.cat([p.grad.reshape(-1) for p in m.parameters()])

    for bn_mode in ("eval", "train"):
        ref = fresh_model().train(bn_mode == "train")
        F.cross_entropy(ref(x_all), y_all).backward()          # single process, full batch
        g_ref = flat_grad(ref)
        for comm in ("manual", "ddp", "ddp_fp16"):
            net = fresh_model().train(bn_mode == "train")
            if comm == "manual":
                F.cross_entropy(net(x_loc), y_loc).backward()
                allreduce_gradients(list(net.parameters()), world_size)
                g = flat_grad(net)
            else:
                ddp = wrap_model(net, device, comm).train(bn_mode == "train")
                F.cross_entropy(ddp(x_loc), y_loc).backward()  # DDP averages during backward
                g = flat_grad(ddp.module)
            rel = ((g - g_ref).norm() / g_ref.norm()).item()
            report["gradient_equivalence"][f"{comm}/bn_{bn_mode}"] = {
                "max_abs_err": (g - g_ref).abs().max().item(), "rel_l2_err": rel}

    # (b) replica consistency after 5 optimiser steps on different, augmented data per rank
    report["replica_max_param_spread"] = {}
    for comm in ("manual", "ddp", "ddp_fp16"):
        torch.manual_seed(args.seed)
        ddp = wrap_model(MODELS[args.model](num_classes=10).to(device), device, comm)
        params = list(ddp.parameters())
        opt = optim.SGD(params, lr=0.05, momentum=0.9)
        g = torch.Generator().manual_seed(1000 + rank)
        for _ in range(5):
            idx = torch.randint(0, len(ytr), (32,), generator=g)
            x = ((xtr[idx].float() / 255 - mean) / std)
            x = torch.where(torch.rand(32, 1, 1, 1, generator=g) < 0.5, x.flip(3), x).to(device)
            opt.zero_grad(set_to_none=True)
            if comm == "manual":
                with ddp.no_sync():
                    F.cross_entropy(ddp(x), ytr[idx].to(device)).backward()
                allreduce_gradients(params, world_size)
            else:
                F.cross_entropy(ddp(x), ytr[idx].to(device)).backward()
            opt.step()
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        hi, lo = flat.clone(), flat.clone()
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        report["replica_max_param_spread"][comm] = (hi - lo).abs().max().item()

    if rank == 0:
        with open(result_file, "w") as f:
            json.dump(report, f, indent=2)
    dist.barrier()
    dist.destroy_process_group()


def check_partitions(n: int, world_size: int, seed: int) -> Dict[str, bool]:
    """(c) the training shards are disjoint & equal-sized; the test split covers every image once."""
    shards = [list(DistributedSampler(range(n), num_replicas=world_size, rank=r, shuffle=True,
                                      seed=seed, drop_last=True)) for r in range(world_size)]
    flat = [i for s in shards for i in s]
    test = [i for r in range(world_size) for i in range(r, n, world_size)]
    return {"train_shards_disjoint": len(flat) == len(set(flat)),
            "train_shards_equal_size": len({len(s) for s in shards}) == 1,
            "train_coverage": len(flat) == world_size * (n // world_size),
            "test_split_exact_cover": sorted(test) == list(range(n))}


def run_verify(cli: argparse.Namespace) -> bool:
    args = resolve_run(cli, dict(tag="verify", workers=cli.verify_workers, device="cpu", threads=1))
    args.fake_data = args.fake_data or not os.path.isdir(os.path.join(args.data_dir, "cifar-10-batches-py"))
    args.train_size = 512
    tolerances = {"manual": 1e-5, "ddp": 1e-5, "ddp_fp16": 5e-3}   # fp16 wire format has ~1e-3 rel. error
    summary: Dict[str, Any] = {"tolerances_rel_l2": tolerances, "runs": {}}
    ok_all = True
    for ws in cli.verify_workers:
        if ws > (psutil.cpu_count(logical=False) or 1):
            continue
        out = os.path.join(args.out_dir, f"verify_w{ws}.json")
        mp.spawn(verify_worker, args=(ws, args, find_free_port(), out), nprocs=ws, join=True)
        with open(out) as f:
            rep = json.load(f)
        os.remove(out)
        rep["partition"] = check_partitions(50000, ws, args.seed)
        checks = {}
        for key, v in rep["gradient_equivalence"].items():
            comm, bn = key.split("/")
            if bn == "bn_eval":   # BN in train mode is *expected* to differ (per-shard batch statistics)
                checks[f"grad_equiv[{key}]"] = v["rel_l2_err"] <= tolerances[comm]
        for comm, spread in rep["replica_max_param_spread"].items():
            checks[f"replicas_identical[{comm}]"] = spread <= 1e-6
        checks.update({f"partition[{k}]": v for k, v in rep["partition"].items()})
        rep["checks"], rep["passed"] = checks, all(checks.values())
        summary["runs"][str(ws)] = rep
        ok_all &= rep["passed"]

        LOG.info("---- verification, %d workers (%s data) ----", ws, "random" if args.fake_data else "CIFAR-10")
        for key, v in rep["gradient_equivalence"].items():
            LOG.info("  grad %-22s rel L2 err %.2e  max abs %.2e", key, v["rel_l2_err"], v["max_abs_err"])
        for comm, spread in rep["replica_max_param_spread"].items():
            LOG.info("  replica param spread after 5 steps [%-8s] %.3e", comm, spread)
        for k, v in checks.items():
            LOG.info("  %-40s %s", k, "PASS" if v else "FAIL")
    summary["passed"] = ok_all
    with open(os.path.join(args.out_dir, "verify.json"), "w") as f:
        json.dump(summary, f, indent=2)
    LOG.info("VERIFICATION %s", "PASSED" if ok_all else "FAILED")
    return ok_all


def commbench_worker(rank: int, world_size: int, args: argparse.Namespace, port: int, result_file: str) -> None:
    """Time an isolated all-reduce for message sizes 1 KB .. 16 MB (+ the model's gradient size)."""
    setup_logging(rank if rank else None)
    device = configure_worker(rank, world_size, args)
    init_distributed(rank, world_size, port, device)
    grad_bytes = sum(p.numel() for p in MODELS[args.model]().parameters()) * 4
    sizes = sorted({4 ** k for k in range(5, 13)} | {grad_bytes})
    rows = []
    for nbytes in sizes:
        buf = torch.ones(nbytes // 4)
        for _ in range(5):                              # warm-up (connection set-up, page faults)
            dist.all_reduce(buf)
        times = []
        for _ in range(30):
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_reduce(buf)
            times.append(time.perf_counter() - t0)
        med, p90 = all_reduce_values([statistics.median(times), float(np.percentile(times, 90))], device,
                                     dist.ReduceOp.MAX)
        rows.append({"bytes": nbytes, "median_ms": 1000 * med, "p90_ms": 1000 * p90})
    if rank == 0:
        with open(result_file, "w") as f:
            json.dump({"world_size": world_size, "grad_bytes": grad_bytes, "rows": rows}, f, indent=2)
    dist.barrier()
    dist.destroy_process_group()


def run_commbench(cli: argparse.Namespace) -> None:
    """alpha-beta fit of t(m) = alpha + 2(N-1)/N * m / beta for each N (CPU workers, gloo)."""
    args = resolve_run(cli, dict(tag="commbench", workers=cli.workers or [2, 4, 8], device="cpu"))
    workers = pick_device_and_workers(args)
    out: Dict[str, Any] = {}
    for ws in [w for w in workers if w > 1]:
        path = os.path.join(args.out_dir, f"_commbench_w{ws}.json")
        mp.spawn(commbench_worker, args=(ws, args, find_free_port(), path), nprocs=ws, join=True)
        with open(path) as f:
            rep = json.load(f)
        os.remove(path)
        x = np.array([2 * (ws - 1) / ws * r["bytes"] for r in rep["rows"]])
        y = np.array([r["median_ms"] / 1000 for r in rep["rows"]])
        slope, alpha = np.polyfit(x, y, 1)
        rep["alpha_ms"], rep["beta_gbs"] = 1000 * alpha, 1 / slope / 1024 ** 3
        rep["grad_allreduce_ms"] = next(r["median_ms"] for r in rep["rows"] if r["bytes"] == rep["grad_bytes"])
        out[str(ws)] = rep
        LOG.info("all-reduce N=%d: alpha = %.3f ms, beta = %.2f GB/s, %.2f MB gradient -> %.1f ms (isolated)",
                 ws, rep["alpha_ms"], rep["beta_gbs"], rep["grad_bytes"] / 1024 ** 2, rep["grad_allreduce_ms"])
    with open(os.path.join(args.out_dir, "commbench.json"), "w") as f:
        json.dump({"threads_per_worker": args.threads, "results": out}, f, indent=2)


# =============================================================================================
# 8. Experiment driver and summary metrics
# =============================================================================================
def run_experiment(cli: argparse.Namespace, preset: Dict[str, Any]) -> List[Dict[str, Any]]:
    args = resolve_run(cli, preset)
    workers = pick_device_and_workers(args)
    if not args.fake_data:
        ensure_cifar10(args.data_dir)
    LOG.info("=== %s | device %s | workers %s | threads/worker %s | epochs %d | global batch %d | lr %.3f | comm %s",
             args.tag, args.device, workers, args.threads if args.device == "cpu" else "-",
             args.epochs, args.batch_size, args.lr, args.comm)
    results = []
    for ws in workers:
        out = os.path.join(args.out_dir, f"{args.tag}_w{ws}.json")
        if args.cooldown:
            time.sleep(args.cooldown)
        t0 = time.time()
        mp.spawn(main_worker, args=(ws, args, find_free_port(), out), nprocs=ws, join=True)
        wall = time.time() - t0
        with open(out) as f:
            res = json.load(f)
        res["wall_time_incl_startup"] = wall
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        results.append(res)
        LOG.info("%s: %d worker(s) finished in %.1f s wall-clock (incl. process start-up)", args.tag, ws, wall)
    return results


def load_results(out_dir: str) -> List[Dict[str, Any]]:
    res = []
    for path in glob.glob(os.path.join(out_dir, "*_w*.json")):
        if os.path.basename(path).startswith("verify"):
            continue
        with open(path) as f:
            res.append(json.load(f))
    return sorted(res, key=lambda r: (r["tag"], r["workers"]))


def _same_workload(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return all(a[k] == b[k] for k in ("train_images", "epochs", "global_batch", "model"))


def find_baseline(r: Dict[str, Any], pool: List[Dict[str, Any]], same_device: bool) -> Optional[Dict[str, Any]]:
    """T1: the 1-worker run with the same workload (prefer the same tag, then manual comm)."""
    cands = [b for b in pool if b["workers"] == 1 and _same_workload(r, b)
             and (not same_device or b["device"] == r["device"])]
    cands.sort(key=lambda b: (b["tag"] != r["tag"], b["comm_mode"] != "manual", b["device"] != r["device"],
                              bool(b.get("amp"))))
    return cands[0] if cands else None


def summarise(results: List[Dict[str, Any]], pool: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Speedup S(N)=T1/TN, efficiency E(N)=S/N, Karp-Flatt serial fraction, comm/wait/data %, gap."""
    pool = pool or results
    rows = []
    for r in results:
        base_t = find_baseline(r, pool, same_device=True)
        base_a = find_baseline(r, pool, same_device=False)
        n, manual = r["workers"], r["comm_mode"] == "manual" and r["workers"] > 1
        s = (base_t["avg_epoch_time_no_warmup"] / r["avg_epoch_time_no_warmup"]) if base_t else None
        kf = ((1 / s - 1 / n) / (1 - 1 / n)) if (s and n > 1) else None
        rows.append({
            "tag": r["tag"], "device": r["device"], "workers": n, "threads_per_worker": r["threads_per_worker"],
            "comm_mode": r["comm_mode"], "global_batch": r["global_batch"], "local_batch": r["local_batch"],
            "train_images": r["train_images"], "epochs": r["epochs"],
            "avg_epoch_time_s": round(r["avg_epoch_time_no_warmup"], 2),
            "epoch_time_cv_pct": round(100 * r.get("epoch_time_cv", 0), 1),
            "total_train_time_s": round(r["total_train_time"], 1),
            "throughput_img_s": round(r["history"][-1]["steps"] * r["global_batch"] / r["avg_epoch_time_no_warmup"], 1),
            "speedup": round(s, 3) if s else "",
            "efficiency_pct": round(100 * s / n, 1) if s else "",
            "karp_flatt": round(kf, 4) if kf is not None else "",
            "compute_pct": round(100 * r["compute_fraction"], 1),
            "data_pct": round(100 * r["data_fraction"], 2),
            "wait_pct": round(100 * r["wait_fraction"], 2) if manual else "",
            "comm_pct": round(100 * r["comm_fraction"], 2) if manual else "",
            "comm_ms_per_step": round(r["comm_per_step_ms"], 2) if manual else "",
            "ring_mb_per_step": round(r["ring_mb_per_step"], 2),
            "bus_bw_gbs": round(r["bus_bandwidth_gbs"], 2) if manual and r["bus_bandwidth_gbs"] else "",
            "compute_imbalance": round(r["compute_imbalance"], 3),
            "step_p50_ms": round(r["step_time_p50_ms"], 1), "step_p95_ms": round(r["step_time_p95_ms"], 1),
            "final_test_acc": round(r["final_test_acc"], 2), "best_test_acc": round(r["best_test_acc"], 2),
            "accuracy_gap": round(base_a["final_test_acc"] - r["final_test_acc"], 2) if base_a else "",
            "peak_rss_mb": round(r["peak_rss_mb_all_workers"], 0),
        })
    return rows


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    LOG.info("%-14s %3s %-8s %6s %9s %8s %7s %7s %7s %7s %8s %6s", "tag", "N", "comm", "batch", "epoch[s]",
             "speedup", "E(N)%", "comm%", "wait%", "acc%", "gap", "KF")
    for r in rows:
        LOG.info("%-14s %3d %-8s %6d %9.2f %8s %7s %7s %7s %7.2f %8s %6s", r["tag"], r["workers"], r["comm_mode"],
                 r["global_batch"], r["avg_epoch_time_s"], r["speedup"], r["efficiency_pct"], r["comm_pct"],
                 r["wait_pct"], r["final_test_acc"], r["accuracy_gap"], r["karp_flatt"])


# =============================================================================================
# 9. Plots (static PNGs for the report)
# =============================================================================================
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
        "lines.markersize": 7, "text.color": INK,
    })
    return plt


def _save(fig, out_dir: str, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_plots(out_dir: str) -> List[str]:
    plt = _style()
    results = load_results(out_dir)
    if not results:
        LOG.info("No results in %s - nothing to plot.", out_dir)
        return []
    rows = summarise(results)
    by_tag: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_tag.setdefault(r["tag"], []).append(r)
    rowmap = {(r["tag"], r["workers"]): r for r in rows}
    made: List[str] = []

    # --- scaling family: manual (scaling) + ddp + ddp_fp16, all relative to the same T1 ---------
    family = [(t, lbl) for t, lbl in (("scaling", "manual all-reduce (timed)"), ("comm_ddp", "DDP overlapped"),
                                      ("comm_ddp_fp16", "DDP + fp16 compression"),
                                      ("scaling_pinned", "manual, cores pinned (P first)")) if t in by_tag]
    if family:
        base = next((r for r in by_tag.get("scaling", []) if r["workers"] == 1), None)
        fig, ax = plt.subplots(figsize=(5.6, 4))
        maxn = 1
        for i, (tag, lbl) in enumerate(family):
            pts = sorted([(r["workers"], rowmap[(tag, r["workers"])]["speedup"]) for r in by_tag[tag]])
            if base and all(n != 1 for n, _ in pts):
                pts = [(1, 1.0)] + pts
            pts = [(n, s) for n, s in pts if s != ""]
            if not pts:
                continue
            maxn = max(maxn, max(n for n, _ in pts))
            ax.plot(*zip(*pts), "o-", color=SERIES[i], label=lbl)
        ns = [1, 2, 4, 8, 16][: [1, 2, 4, 8, 16].index(maxn) + 1] if maxn in (1, 2, 4, 8, 16) else list(range(1, maxn + 1))
        ax.plot(ns, ns, "--", color=INK2, linewidth=1.2, label="ideal (linear)")
        ax.set_xticks(ns)
        ax.set(xlabel="Number of workers N", ylabel="Speedup S(N) = T1 / TN", title="Speedup vs number of workers")
        ax.legend()
        _save(fig, out_dir, "fig_speedup.png"); made.append("fig_speedup.png")

        fig, ax = plt.subplots(figsize=(6.2, 4))
        width = 0.8 / len(family)
        all_n = sorted({r["workers"] for t, _ in family for r in by_tag[t]} - {1})   # E(1) = 100 % by definition
        for i, (tag, lbl) in enumerate(family):
            effs = [rowmap.get((tag, n), {}).get("efficiency_pct", "") for n in all_n]
            xs = [k + (i - (len(family) - 1) / 2) * width for k in range(len(all_n))]
            vals = [e if e != "" else 0 for e in effs]
            bars = ax.bar(xs, vals, width, color=SERIES[i], edgecolor="white", linewidth=2, label=lbl)
            for b, e in zip(bars, effs):
                if e != "" and len(family) == 1:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{e:.0f}%", ha="center",
                            fontsize=8, color=INK2)
        ax.axhline(100, color=INK2, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(all_n)), [str(n) for n in all_n])
        ax.set(xlabel="Number of workers N", ylabel="Parallel efficiency E(N) (%)", title="Parallel efficiency",
               ylim=(0, 115))
        ax.legend(loc="upper right", fontsize=8)
        _save(fig, out_dir, "fig_efficiency.png"); made.append("fig_efficiency.png")

    if "scaling" in by_tag:
        sc = sorted(by_tag["scaling"], key=lambda r: r["workers"])
        labels = [str(r["workers"]) for r in sc]
        avg = lambda r, k: sum(h[k] for h in r["history"][1:] or r["history"]) / max(len(r["history"][1:]), 1)  # noqa
        parts = [("compute (fwd + bwd + update)", "compute_time"), ("data loading", "data_time"),
                 ("waiting for slowest worker", "wait_time"), ("gradient all-reduce", "comm_time")]
        fig, ax = plt.subplots(figsize=(6.2, 4))
        bottom = [0.0] * len(sc)
        for i, (lbl, key) in enumerate(parts):
            vals = [avg(r, key) for r in sc]
            ax.bar(labels, vals, 0.6, bottom=bottom, color=SERIES[i], edgecolor="white", linewidth=2, label=lbl)
            bottom = [b + v for b, v in zip(bottom, vals)]
        for k, r in enumerate(sc):
            ax.text(k, bottom[k] * 1.01, f"{r['avg_epoch_time_no_warmup']:.0f}s", ha="center", va="bottom",
                    fontsize=8, color=INK2)
        ax.set(xlabel="Number of workers N", ylabel="Time per epoch (s, warm-up excluded)",
               title="Where the epoch time goes (manual mode)")
        ax.legend(fontsize=8)
        _save(fig, out_dir, "fig_time_breakdown.png"); made.append("fig_time_breakdown.png")

        multi = [r for r in sc if r["workers"] > 1]
        if multi:
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
            ns = [r["workers"] for r in multi]
            ax[0].plot(ns, [100 * r["comm_fraction"] for r in multi], "o-", color=SERIES[0], label="all-reduce")
            ax[0].plot(ns, [100 * (r["comm_fraction"] + r["wait_fraction"]) for r in multi], "s--", color=SERIES[1],
                       label="all-reduce + waiting")
            ax[0].set(xlabel="Number of workers N", ylabel="% of training time", title="Communication overhead")
            ax[0].set_xticks(ns); ax[0].legend(fontsize=8); ax[0].set_ylim(bottom=0)
            ax[1].plot(ns, [r["comm_per_step_ms"] for r in multi], "o-", color=SERIES[0])
            for n, r in zip(ns, multi):
                ax[1].annotate(f"{r['ring_mb_per_step']:.1f} MB", (n, r["comm_per_step_ms"]),
                               textcoords="offset points", xytext=(10, -12), ha="left", fontsize=8, color=INK2)
            ax[1].set_ylim(0, 1.2 * max(r["comm_per_step_ms"] for r in multi))
            ax[1].set_xlim(min(ns) - 0.5, max(ns) + 1.5)
            ax[1].set(xlabel="Number of workers N", ylabel="All-reduce time per step (ms)",
                      title="All-reduce time per step (label: MB sent)")
            ax[1].set_xticks(ns); ax[1].set_ylim(bottom=0)
            _save(fig, out_dir, "fig_comm_overhead.png"); made.append("fig_comm_overhead.png")

            kf = [(r["workers"], rowmap[("scaling", r["workers"])]["karp_flatt"]) for r in multi]
            kf = [(n, e) for n, e in kf if e != ""]
            if kf:
                fig, ax = plt.subplots(figsize=(5, 3.6))
                ax.plot(*zip(*kf), "o-", color=SERIES[0])
                ax.axhline(0, color=INK2, linewidth=1)
                ax.set(xlabel="Number of workers N", ylabel="Experimental serial fraction e",
                       title="Karp-Flatt metric (rising e = growing overhead)")
                ax.set_xticks([n for n, _ in kf])
                _save(fig, out_dir, "fig_karp_flatt.png"); made.append("fig_karp_flatt.png")

    # --- compute inflation: CPU-seconds of fwd/bwd/update per epoch summed over workers ---------
    def _compute(r):
        later = r["history"][1:] or r["history"]
        return sum(h["compute_time"] for h in later) / len(later)

    infl = [(tag, lbl) for tag, lbl in (("scaling", "OS-scheduled cores"), ("scaling_pinned", "pinned, P-cores first"))
            if any(r["workers"] == 1 for r in by_tag.get(tag, []))]
    if infl:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ns = sorted({r["workers"] for t, _ in infl for r in by_tag[t]})
        width = 0.8 / len(infl)
        for i, (tag, lbl) in enumerate(infl):
            runs = {r["workers"]: r for r in by_tag[tag]}
            vals = [n * _compute(runs[n]) / _compute(runs[1]) if n in runs else 0 for n in ns]
            xs = [k + (i - (len(infl) - 1) / 2) * width for k in range(len(ns))]
            bars = ax.bar(xs, vals, width, color=SERIES[i], edgecolor="white", linewidth=2, label=lbl)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK2, padding=2)
        ax.axhline(1, color=INK2, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set(xlabel="Number of workers N", ylabel="N x compute(N) / compute(1)",
               title="Compute inflation (1.0 = no per-worker slowdown)")
        ax.legend(fontsize=8)
        _save(fig, out_dir, "fig_compute_inflation.png"); made.append("fig_compute_inflation.png")

    # --- convergence: test accuracy vs wall-clock training time ---------------------------------
    sc_runs = by_tag.get("scaling", [])
    sc_desc = (f"{sc_runs[0]['train_images'] // 1000}k images, {sc_runs[0]['epochs']} epochs" if sc_runs else "")
    for tags, name, title in ((["scaling"], "fig_accuracy_vs_time.png", f"Test accuracy vs training time ({sc_desc})"),
                              (["final_cpu", "final_gpu", "final_gpu_bf16"], "fig_final_accuracy_vs_time.png",
                               "Test accuracy vs training time (full CIFAR-10)")):
        runs = [r for t in tags for r in sorted(by_tag.get(t, []), key=lambda r: r["workers"])]
        if not runs:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, r in enumerate(runs):
            t = [0] + [h["train_time_so_far"] / 60 for h in r["history"]]
            a = [10.0] + [h["test_acc"] for h in r["history"]]   # untrained model = 10% (chance)
            ax.plot(t, a, "o-", color=SERIES[i], markersize=5,
                    label=f"{r['workers']} {r['device'].upper()} worker{'s' if r['workers'] > 1 else ''}"
                          + (" (bf16)" if r.get("amp") else ""))
        ax.set(xlabel="Training time (minutes)", ylabel="Test accuracy (%)", title=title)
        ax.legend(fontsize=8)
        _save(fig, out_dir, name); made.append(name)

    final_runs = [r for t in ("final_cpu", "final_gpu", "final_gpu_bf16") for r in by_tag.get(t, [])]
    if final_runs:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for i, r in enumerate(final_runs):
            ep = [h["epoch"] for h in r["history"]]
            lbl = f"{r['workers']} {r['device'].upper()}" + (" bf16" if r.get("amp") else "")
            ax[0].plot(ep, [h["train_loss"] for h in r["history"]], "-", color=SERIES[2 * i], label=f"train, {lbl}")
            ax[0].plot(ep, [h["test_loss"] for h in r["history"]], "--", color=SERIES[2 * i + 1], label=f"test, {lbl}")
            ax[1].plot(ep, [h["train_acc"] for h in r["history"]], "-", color=SERIES[2 * i], label=f"train, {lbl}")
            ax[1].plot(ep, [h["test_acc"] for h in r["history"]], "--", color=SERIES[2 * i + 1], label=f"test, {lbl}")
        ax[0].set(xlabel="Epoch", ylabel="Loss (train incl. label smoothing)", title="Loss per epoch")
        ax[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy per epoch")
        for a in ax:
            a.legend(fontsize=8)
        _save(fig, out_dir, "fig_final_curves.png"); made.append("fig_final_curves.png")

    # --- global batch sweep: efficiency and accuracy (two panels, never a dual axis) -----------
    batch_rows = [r for r in rows if r["tag"] in ("scaling", "batch_b1024") and r["workers"] in (1, 8)]
    b8 = sorted([r for r in batch_rows if r["workers"] == 8], key=lambda r: r["global_batch"])
    if len(b8) > 1:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        lbls = [str(r["global_batch"]) for r in b8]
        ax[0].bar(lbls, [r["efficiency_pct"] or 0 for r in b8], 0.5, color=SERIES[0], edgecolor="white", linewidth=2)
        ax[0].set(xlabel="Global batch size (8 workers)", ylabel="Parallel efficiency E(8) (%)",
                  title="Efficiency vs global batch")
        b1 = {r["global_batch"]: r for r in batch_rows if r["workers"] == 1}
        width = 0.35
        xs = range(len(b8))
        ax[1].bar([x - width / 2 for x in xs], [b1.get(r["global_batch"], {}).get("final_test_acc", 0) for r in b8],
                  width, color=SERIES[0], edgecolor="white", linewidth=2, label="1 worker")
        ax[1].bar([x + width / 2 for x in xs], [r["final_test_acc"] for r in b8], width, color=SERIES[1],
                  edgecolor="white", linewidth=2, label="8 workers")
        ax[1].set_xticks(list(xs), lbls)
        ax[1].set(xlabel="Global batch size", ylabel=f"Test accuracy after {b8[0]['epochs']} epochs (%)",
                  title="Accuracy vs global batch", ylim=(0, 100))
        ax[1].legend(fontsize=8, loc="upper right", ncols=2)
        for a in ax:
            for c in a.containers:
                a.bar_label(c, fmt="%.1f", fontsize=8, color=INK2, padding=2)
        _save(fig, out_dir, "fig_batch_sweep.png"); made.append("fig_batch_sweep.png")

    # --- isolated all-reduce micro-benchmark vs in-training all-reduce -------------------------
    cb_path = os.path.join(out_dir, "commbench.json")
    if os.path.exists(cb_path):
        with open(cb_path) as f:
            cb = json.load(f)["results"]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for i, (ws, rep) in enumerate(sorted(cb.items(), key=lambda kv: int(kv[0]))):
            xs = [r["bytes"] / 1024 ** 2 for r in rep["rows"]]
            ax.plot(xs, [r["median_ms"] for r in rep["rows"]], "o-", color=SERIES[i], markersize=5,
                    label=f"N={ws}: isolated (alpha {rep['alpha_ms']:.2f} ms, beta {rep['beta_gbs']:.2f} GB/s)")
            run = next((r for r in by_tag.get("scaling", []) if r["workers"] == int(ws)), None)
            if run:
                ax.plot([rep["grad_bytes"] / 1024 ** 2], [run["comm_per_step_ms"]], "D", color=SERIES[i],
                        markersize=9, markeredgecolor=INK, label=f"N={ws}: measured during training")
        ax.set(xscale="log", yscale="log", xlabel="Message size (MB)", ylabel="All-reduce time (ms, median)",
               title="gloo all-reduce cost: isolated vs during training")
        ax.legend(fontsize=7)
        _save(fig, out_dir, "fig_commbench.png"); made.append("fig_commbench.png")

    write_csv(rows, os.path.join(out_dir, "summary_all.csv"))
    LOG.info("Wrote %d figures and summary_all.csv to %s", len(made), out_dir)
    return made


# =============================================================================================
# 10. Entry point
# =============================================================================================
def main() -> None:
    setup_logging()
    cli = build_parser().parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    if sys.platform == "win32":
        # some Windows wheels are built without libuv for the TCPStore rendezvous
        os.environ.setdefault("USE_LIBUV", "0")
    elif "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"

    LOG.info("Machine: %s | OS power throttling disabled: %s", json.dumps(machine_info()),
             disable_os_power_throttling())
    commands = PIPELINE if cli.command == "all" else [cli.command]
    t_start = time.time()
    for cmd in commands:
        if cmd == "verify":
            if not run_verify(cli):
                sys.exit("Verification failed - fix before running experiments.")
        elif cmd == "commbench":
            run_commbench(cli)
        elif cmd == "plot":
            make_plots(cli.out_dir)
        else:
            if cmd == "final_gpu" and not torch.cuda.is_available():
                LOG.info("No CUDA GPU - skipping final_gpu.")
                continue
            results = []
            for preset in EXPERIMENTS[cmd]:
                results += run_experiment(cli, preset)
            rows = summarise(results, pool=results + load_results(cli.out_dir))
            write_csv(rows, os.path.join(cli.out_dir, f"{cmd}_summary.csv"))
            print_summary(rows)
    LOG.info("Done in %.1f s.", time.time() - t_start)


if __name__ == "__main__":
    main()
