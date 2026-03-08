import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from dmicc.model.npc_model import NonParametricClassifier
from dmicc.tools.normalize import Normalize
from dmicc.tools.tools import check_clustering_metrics
from dmicc.losses.Loss_IMI import crossview_contrastive_Loss
from dmicc.losses.Loss_ID import Loss_ID
from dmicc.losses.Loss_FMI import Loss_FMI
import random
from datetime import datetime
import time



class LabeledCSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):

        self.data = pd.read_csv(csv_path, header=None)

        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values

        self.mean = torch.FloatTensor(self.features.mean(axis=0))
        self.std = torch.FloatTensor(self.features.std(axis=0) + 1e-8)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        label = self.labels[idx]
        x = (torch.FloatTensor(x) - self.mean) / self.std

        if self.transform:
            x1 = self.transform(x)
            x2 = self.transform(x)
        else:
            x1, x2 = x, x

        return x1, x2, torch.tensor(label, dtype=torch.long), idx 


class MLP(nn.Module):

    def __init__(self, input_dim, low_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, low_dim)
        )

    def forward(self, x):
        return self.net(x)

def create_unique_dir(seed, base_path="dmicc/checkpoint/frog/o"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_path, f"seed_{seed}", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to labeled CSV file")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--input_dim", type=int, required=True, help="Number of feature columns")
    parser.add_argument("--gpus", type=str, default="0")
    return parser.parse_args()


def main(seed):

    set_seed(seed)
    save_dir = create_unique_dir(seed)
    csv_path = f'dmicc/data/frog/frog_o+label_{seed}.csv'
    input_dim = 22
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    epochs = 20


    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x * (1 + 0.1 * torch.randn_like(x))),
        transforms.Lambda(lambda x: torch.clamp(x, min=-3, max=3))
    ])

    train_data = LabeledCSVDataset(csv_path, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)

    # train_loader = DataLoader(train_data, batch_size=batch_size,
    #                           shuffle=True, num_workers=0, pin_memory=True, drop_last=True,)

    net = MLP(input_dim=input_dim).to(device)
    norm = Normalize(2).to(device)
    npc = NonParametricClassifier(input_dim=128, output_dim=len(train_data),
                                  tau=1.0, momentum=0.5).to(device)
    criterion = {
        "id": Loss_ID(tau2=2.0).to(device),
        "fmi": Loss_FMI().to(device)
    }

    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

    ##results = {'epoch': [], 'loss': [], 'acc': [], 'nmi': [], 'ari': []}
    # results = {'epoch': [], 'loss': [], 'sil': [], 'ch': [], 'db': [], 'acc': [], 'nmi': [], 'ari':  []}
    results = {'epoch': [], 'loss': [], 'acc': [], 'ari': [], 'ch': [], 'db': [], 'nmi': [], 'sil': []}

    total_start = time.time()  # ✅ 整体计时开始
    for epoch in range(epochs):
        epoch_start = time.time()  # ✅ 每个 epoch 开始计时
        net.train()
        for x1, x2, _, idx in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x1, x2, idx = x1.to(device), x2.to(device), idx.to(device)


            f1, f2 = norm(net(x1)), norm(net(x2))
            loss = criterion["id"](npc(f1, idx), idx) + 1e-5 * criterion["fmi"](f1) + 1e-6 * crossview_contrastive_Loss(
                f1, f2)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # 每个 epoch 结束时输出耗时
        epoch_time = time.time() - epoch_start
        # print(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

        if (epoch + 1) % 10 == 0:
            sil, ch, db, acc, nmi, ari = check_clustering_metrics(npc, train_loader)
            results['epoch'].append(epoch + 1)
            results['loss'].append(loss.item())
            results['sil'].append(sil)
            results['ch'].append(ch)
            results['db'].append(db)
            results['acc'].append(acc)
            results['nmi'].append(nmi)
            results['ari'].append(ari)

            print(f"Epoch {epoch + 1}: "
                  f"Loss={loss.item():.4f}, "
                  f"sil={sil:.4f}, "
                  f"CH={ch:.4f}, "
                  f"DB={db:.4f},"
                  f"ACC={acc:.4f}, "
                  f"NMI={nmi:.4f}, "
                  f"ARI={ari:.4f}")


            ##torch.save({
                ##'model': net.state_dict(),
                ##'optimizer': optimizer.state_dict(),
                ##'epoch': epoch + 1
            ##}, f'checkpoint_epoch{epoch + 1}.pth')
            torch.save({
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch + 1}.pth'))

    total_time = time.time() - total_start
    print(f"\n🏁 Training completed in {total_time / 60:.2f} minutes")

    folder = f'dmicc/result/frog/o'
    file_path = f'{folder}/frog_dmicc_o_seed{seed}.csv'
    # 创建文件夹（如果不存在）
    os.makedirs(folder, exist_ok=True)
    # 保存 CSV
    pd.DataFrame(results).to_csv(file_path, index=False)

    # pd.DataFrame(results).to_csv('dmicc.csv', index=False)
    # pd.DataFrame(results).to_csv(f'./dmicc/result/frog/h/frog_dmicc_h_seed{seed}.csv', index=False)


if __name__ == "__main__":
    # seed = 43

    for seed in range (42, 52):
        print("seed:", seed)
        main(seed)