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


    # load weight
    LOAD_ENV_PATH = os.path.join(model_load_path, 'epoch_{}.pth'.format(starting_epoch_idx))
    model = RawNet(parser1['model'])
    model.load_state_dict(torch.load(LOAD_ENV_PATH, weights_only=True))
    print("Load model from " + LOAD_ENV_PATH)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    # save weight
    torch.save({
        'epoch': starting_epoch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 100000,
        }, os.path.join(model_load_path, f'epoch_{starting_epoch_idx}_new.pth'))
