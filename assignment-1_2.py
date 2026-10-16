import os
import psutil
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset

program_start_time = time.time()
# ----------------------------------
# Basic Residual Block
# ----------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
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
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out = out + identity
        return F.relu(out, inplace=True)


# ----------------------------------
# ResNet-18
# ----------------------------------
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()

        self.in_channels = 32

        # Initial convolution for CIFAR-10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet10(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="gloo",rank=rank,world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def run_epoch(model, loader, criterion, device, optimizer=None):
    """Runs one pass over `loader`. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def main(rank, world_size):
    # ---- Device ----
    device = torch.device("cpu")
    #print("Using device:", device)

    setup(rank, world_size)

    # ---- Data ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    train_subset = Subset(train_dataset, range(10000))
    test_subset  = Subset(test_dataset, range(2000))

    if rank == 0:

        print("Number of training images:", len(train_subset))
        #print("Raw training data shape:", train_subset.data.shape)  # (50000, 32, 32, 3)
        print("Number of test images:", len(test_subset))
        #print("Raw test data shape:", test_subset.data.shape)       # (10000, 32, 32, 3)

    pin = device.type == "cuda"  # pin_memory is only useful (and supported) for CUDA
    #train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,num_workers=2, pin_memory=pin)
    #test_loader = DataLoader(test_subset, batch_size=128, shuffle=False,num_workers=2, pin_memory=pin)

    train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True)

    train_loader = DataLoader(train_dataset,batch_size=128,sampler=train_sampler,num_workers=2)

    test_sampler = DistributedSampler(test_dataset,num_replicas=world_size,rank=rank,shuffle=False)

    test_loader = DataLoader(test_dataset,batch_size=128,sampler=test_sampler,num_workers=2)

    # ---- Model / loss / optimizer ----
    model = ResNet10(num_classes=10).to(device)
    model = DDP(model)
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel()for p in model.parameters()if p.requires_grad)

    model_size_mb = (total_params * 4) / (1024 ** 2)
    
    if rank == 0:
        print("\n========== MODEL COMPLEXITY ==========")
        print(f"Total Parameters : {total_params:,}")
        print(f"Trainable Params : {trainable_params:,}")
        print(f"Model Size       : {model_size_mb:.2f} MB")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # ---- Training ----
    num_epochs = 10
    best_acc = 0.0
    process = psutil.Process(os.getpid())
    
    if rank == 0:
        memory_before = (process.memory_info().rss/ 1024 / 1024)

        print(f"Memory Before Training: "f"{memory_before:.2f} MB")

    training_start_time = time.time()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        avg_train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        avg_test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

        scheduler.step()
    training_end_time = time.time()
    training_time = (training_end_time -training_start_time)

    if rank == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {avg_test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
            )
    if rank == 0:
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_resnet18_cifar10.pth")

            print(f"\nBest Test Accuracy: {best_acc:.2f}%")

"""
#if __name__ == "__main__":
    #main()
if __name__ == "__main__":

    world_size = min(4, os.cpu_count())
    #print("Number of CPUs: ", world_size)

    mp.spawn(main,args=(world_size,),nprocs=world_size,join=True )

    program_end_time = time.time()

    print("\n========== EXECUTION SUMMARY ==========")
    print(f"End-to-End Execution Time: "f"{program_end_time - program_start_time:.2f} seconds")
    print(f"Training Time: "f"{training_end_time - training_start_time:.2f} seconds")

"""

if __name__ == "__main__":

    worker_configs = [2, 4, 6, 8]

    results = []

    for world_size in worker_configs:

        if world_size > os.cpu_count():
            continue

        print(f"\n{'='*60}")
        print(f"Running DDP with {world_size} workers")
        print(f"{'='*60}")

        run_start_time = time.time()

        mp.spawn(main,args=(world_size,),nprocs=world_size,join=True)

        run_end_time = time.time()

        execution_time = run_end_time - run_start_time

        results.append({"workers": world_size,"time": execution_time})

        print(f"Execution Time ({world_size} workers): "f"{execution_time:.2f} seconds")

    print("\n")
    print("=" * 60)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 60)

    baseline_time = results[0]["time"]

    for result in results:

        workers = result["workers"]
        exec_time = result["time"]

        speedup = baseline_time / exec_time

        efficiency = speedup / workers

        print(
            f"Workers: {workers:<2} | "
            f"Time: {exec_time:>8.2f} s | "
            f"Speedup: {speedup:.2f} | "
            f"Efficiency: {efficiency*100:.2f}%"
        )