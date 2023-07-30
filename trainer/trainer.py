import termcolor
import torch
import torch.optim as optim
import tqdm
from torch.nn import MSELoss
from models.lstm import LSTM
from dataset.solana_dataset import SolanaDataset
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import pandas as pd
from uitls.utils import AverageMeter


class Trainer:
    def __init__(self, model_config, data_config, train_config):
        self.csv_file = data_config['csv_file']
        self.data = pd.read_csv(self.csv_file)
        train_len = int(0.8 * len(self.data))
        val_len = int(1. * len(self.data))
        self.train_data = self.data.iloc[:train_len]
        self.val_data = self.data.iloc[train_len:val_len]
        self.trainset = SolanaDataset(data=self.train_data,
                                      time_span=data_config['time_span'],
                                      features=data_config['features'], only_last=data_config['only_last'])
        self.valset = SolanaDataset(data=self.val_data,
                                    min=self.trainset.min,
                                    max=self.trainset.max,
                                    time_span=data_config['time_span'],
                                    features=data_config['features'], only_last=data_config['only_last'])
        print(termcolor.colored(f"train length: {len(self.trainset)}, test length: {len(self.valset)}", 'red'))
        self.train_loader = DataLoader(self.trainset,
                                       batch_size=train_config['batch_size'],
                                       shuffle=True,
                                       num_workers=train_config['num_workers'])
        self.val_loader = DataLoader(self.valset,
                                     batch_size=train_config['batch_size'],
                                     shuffle=False,
                                     num_workers=train_config['num_workers']
                                     )
        self.model = LSTM(**model_config)
        self.model = DataParallel(self.model).to(train_config['device'])
        self.criterion = MSELoss()
        self.optimizer = optim.AdamW(params=self.model.parameters(),
                                     lr=train_config['lr'],
                                     weight_decay=train_config['wd'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                 T_max=train_config['epoch'] * len(self.train_loader),
                                                                 eta_min=1e-3 * train_config['lr'])
        self.max_epoch = train_config['epoch']
        self.device = train_config['device']

    def train_epoch(self, epoch):
        pbar = tqdm.tqdm(self.train_loader, desc=f"Training epoch {epoch}/{self.max_epoch}")
        self.model.train()
        loss_meter = AverageMeter()
        for source, target, last in pbar:
            self.optimizer.zero_grad()
            source = source.to(self.device)
            target = target.to(self.device)
            last = last.to(self.device)
            prediction = self.model(source)
            last_close = last[:, 3]
            current_open = prediction[:, 0]
            loss = self.criterion(prediction[:, :-1], target[:, :-1])
            loss_2 = self.criterion(last_close, current_open)
            (loss + loss.item() / loss_2.item() * loss_2 * 50).backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            loss_meter.update(loss.item())
            pbar.set_postfix({"loss": loss_meter.average()})

    def eval_epoch(self, epoch):
        pbar = tqdm.tqdm(self.val_loader, desc=f"Valid epoch {epoch}/{self.max_epoch}")
        self.model.eval()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for source, target, last in pbar:
                self.optimizer.zero_grad()
                source = source.to(self.device)
                target = target.to(self.device)

                prediction = self.model(source)
                loss = self.criterion(prediction[:, :-1], target[:, :-1])
                # loss.backward()
                # self.optimizer.step()
                # self.lr_scheduler.step()
                loss_meter.update(loss.item())
                pbar.set_postfix({"loss": loss_meter.average()})
        return loss_meter.average()

    def train(self):
        best_score = 1e9
        for epoch in range(self.max_epoch):
            self.train_epoch(epoch)
            result = self.eval_epoch(epoch)
            if result < best_score:
                best_score = result
                torch.save(self.model.module.state_dict(), 'best_6.pt')
                print("---> save checkpoint best")
