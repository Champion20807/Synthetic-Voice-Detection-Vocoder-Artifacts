import os
import argparse
import yaml
from model import RawNet
import torch


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/your/path/to/LibriSeVoc/')
    parser.add_argument('--model_load_path', type=str, default='/your/path/to/models')
    parser.add_argument('--model_save_path', type=str, default='/your/path/to/models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    args = parser.parse_args()

    data_path = args.data_path
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay


    # load model config
    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    starting_epoch_idx = int(input())

    # load cuda
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    
    # load weight
    LOAD_ENV_PATH = os.path.join(model_load_path, 'epoch_{}_new.pth'.format(starting_epoch_idx))

    model = RawNet(parser1['model'],device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    print("Load model from " + LOAD_ENV_PATH)
    checkpoint = torch.load(LOAD_ENV_PATH, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(model.block0[0].conv1.weight)
