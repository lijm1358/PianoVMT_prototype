import argparse
from Video2Roll_dataset import Video2RollDataset
from torch.utils.data import DataLoader
import torch
from torch import optim

import Video2RollNet

from Video2Roll_solver import Solver
import torch.nn as nn
from balance_data import MultilabelBalancedRandomSampler


def main(args):
    # train_dataset = Video2RollDataset(subset='train')
    train_dataset = Video2RollDataset(args.image_path, label_root=args.label_path, subset='train', min_key=args.min_key, max_key=args.max_key)
    train_sampler = MultilabelBalancedRandomSampler(train_dataset.train_labels)
    train_data_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=8)
    test_dataset = Video2RollDataset(args.image_path, label_root=args.label_path, subset='test', min_key=args.min_key, max_key=args.max_key)
    test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)
    device = torch.device('cuda')

    net = Video2RollNet.resnet18(num_classes=80-5+1)
    # net = Video2RollNet.resnet18()
    net.load_state_dict(torch.load("./models/Video2Roll_finetune_0.52.pth", map_location="cuda:0"))
    net.fc = nn.Linear(128, args.max_key-args.min_key+1)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    solver = Solver(train_data_loader, test_data_loader, net, criterion, optimizer, scheduler, epochs=50)
    solver.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_path", type=str, default="../PianoVMT_prototype/ytdataset/images")
    parser.add_argument("--label_path", type=str, default="../PianoVMT_prototype/ytdataset/labels_audeo")
    parser.add_argument("--min_key", type=int, default=5)
    parser.add_argument("--max_key", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    
    main(args)
