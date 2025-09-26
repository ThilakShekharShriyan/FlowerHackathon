"""SOCOFing task (dataset/model/train/eval) for Flower."""

import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

# ---------------------------
# Repro (optional)
# ---------------------------

import torch

def _dl_opts():
    use_cuda = torch.cuda.is_available()
    # pin_memory only helps (and is supported) on CUDA
    return dict(
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
    )

def set_repro(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Model
# ---------------------------
class Net(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)  # 96x96 -> 48x48
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 48x48 -> 24x24
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 24x24 -> 12x12
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------------
# Dataset
# ---------------------------
LabelMode = Literal["binary", "fourclass"]

@dataclass
class Sample:
    path: str
    label: int

_TRANSFORMS = Compose(
    [Grayscale(num_output_channels=1), Resize((96, 96)), ToTensor(), Normalize([0.5], [0.5])]
)

# caches
_SCAN_CACHE: Dict[Tuple[str, str], List[Sample]] = {}
_SPLIT_CACHE: Dict[Tuple[int, int, int], List[List[int]]] = {}

_EXTS = ("*.png", "*.PNG", "*.bmp", "*.BMP", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")

def _glob_many(base: str, *segs: str) -> List[str]:
    path = os.path.join(base, *segs)
    out: List[str] = []
    for pat in _EXTS:
        out.extend(glob.glob(os.path.join(path, pat), recursive=True))
    return out

def _scan_files(data_root: str, label_mode: str) -> List[Sample]:
    key = (data_root, label_mode)
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]

    real = _glob_many(data_root, "SOCOFing", "Real", "**")
    alt_easy = _glob_many(data_root, "SOCOFing", "Altered", "Altered-Easy", "**")
    alt_med  = _glob_many(data_root, "SOCOFing", "Altered", "Altered-Medium", "**")
    alt_hard = _glob_many(data_root, "SOCOFing", "Altered", "Altered-Hard", "**")
    altered = alt_easy + alt_med + alt_hard

    samples: List[Sample] = []
    if label_mode == "binary":
        samples += [Sample(p, 0) for p in real]
        samples += [Sample(p, 1) for p in altered]
    else:
        def map_fourclass(pth: str) -> int:
            f = os.path.basename(pth).lower()
            if "obl" in f: return 1
            if "cr" in f or "central_rot" in f or "centralrotation" in f: return 2
            if "zcut" in f or "z_cut" in f: return 3
            return 1
        samples += [Sample(p, 0) for p in real]
        samples += [Sample(p, map_fourclass(p)) for p in altered]

    rng = np.random.default_rng(42)
    rng.shuffle(samples)

    if len(samples) == 0:
        raise ValueError(
            "SOCOFing scan found 0 images. "
            f"Checked under: {os.path.join(data_root, 'SOCOFing')} "
            f"with extensions {list(_EXTS)}. "
            "Fix data-root or directory layout."
        )

    _SCAN_CACHE[key] = samples
    return samples

class SocofingDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("L")
        img = _TRANSFORMS(img)
        return {"img": img, "label": torch.tensor(s.label, dtype=torch.long)}

def _iid_splits(n: int, num_partitions: int, seed: int = 42) -> List[List[int]]:
    key = (n, num_partitions, seed)
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]
    idx = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    parts = np.array_split(idx, num_partitions)
    out = [p.tolist() for p in parts]
    _SPLIT_CACHE[key] = out
    return out

def load_data(
    partition_id: int,
    num_partitions: int,
    data_root: str,
    label_mode: LabelMode = "binary",
    batch_size: int = 32,
):
    # scan
    samples = _scan_files(data_root, label_mode)
    n = len(samples)

    # never create more partitions than samples
    effective = max(1, min(num_partitions, n))
    pid = partition_id % effective
    splits = _iid_splits(n, effective, seed=42)
    part_idx = splits[pid]

    # if something still went wrong, pick the largest non-empty split
    if len(part_idx) == 0:
        lengths = [len(s) for s in splits]
        pid = int(np.argmax(lengths))
        part_idx = splits[pid]

    # 80/20 split (ensure both sides non-empty when possible)
    if len(part_idx) >= 2:
        cut = max(1, min(len(part_idx) - 1, int(0.8 * len(part_idx))))
        train_idx, val_idx = part_idx[:cut], part_idx[cut:]
    else:
        train_idx, val_idx = part_idx, part_idx  # tiny partition: use same sample(s)

    train_ds = SocofingDataset([samples[i] for i in train_idx])
    val_ds   = SocofingDataset([samples[i] for i in val_idx])

    # small partitions: avoid persistent_workers
    nw = 0 if len(part_idx) < 64 else 2
    opts = _dl_opts()
    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        **opts
    )
    valloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        **opts
    )
    return trainloader, valloader

# ---------------------------
# Train / Eval
# ---------------------------
def train(net: nn.Module, trainloader, epochs: int, lr: float, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    total_loss, steps = 0.0, 0
    for _ in range(int(epochs)):
        for batch in trainloader:
            imgs = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(net(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)

def test(net: nn.Module, testloader, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    net.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            imgs = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            out = net(imgs)
            loss_sum += criterion(out, labels).item()
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return loss_sum / max(len(testloader), 1), correct / max(total, 1)
