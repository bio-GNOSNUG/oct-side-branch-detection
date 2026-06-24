import argparse
import yaml
import datetime
import random
import torch
import numpy as np
import os
from torchinfo import summary
from src.models.load_model import load_model
from src.datasets.load_dataset import load_dataset
from src.detection_utils.visualize import plot_loss
from src.detection_utils.engine import train_one_epoch, evaluate, evaluate_loss


def main(config):

    seed = config['SEED']
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('connected to device: {}'.format(device))

    # load dataset
    (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset) = load_dataset(config)

    # get the model using our helper function
    model = load_model(config, device)

    #summary(model, (config['TRAIN_BATCH_SIZE'], config['INPUT_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config['LR'])

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs -- DISABLED
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    save_folder = os.path.join('tests', "results", config['RUN_ID'])
    os.makedirs(save_folder, exist_ok=True)
    print(
    f"Results will be saved to: "
    f"{os.path.abspath(save_folder)}"
)

    best_val_loss = np.inf
    best_val_map = 0
    train_losses = []
    val_losses = []
    val_map_all = []

    for epoch in range(config['EPOCHS']):
        # training for one epoch
        model, metrics, train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=config['TRAIN_PRINT'])
        # update the learning rate
        lr_scheduler.step()
        # validation loss
        val_loss = evaluate_loss(model, val_loader, device=device)

        # evaluate on the test dataset
        evaluator = evaluate(model, val_loader, device=device)

        val_map = evaluator.coco_eval['bbox'].stats[0]

        train_losses.append(train_loss)
        val_losses.append(val_loss.cpu().numpy())
        val_map_all.append(val_map)

        print('Val mAP: {:.3f}, loss: {:.3f}'.format(val_map, val_loss))

        # save model if the performance improves.
        # if val_loss < best_val_loss:
        #     print('Val loss improved from {:.3f} to {:.3f}. Saving model to..{}'.format(best_val_loss, val_loss, save_folder))
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), os.path.join(save_folder, 'best_loss.pt'))
        #     print("saved model new best loss")

        if val_map > best_val_map:
            print('Val mAP improved from {:.3f} to {:.3f}. Saving model to..{}'.format(best_val_map, val_map, save_folder))
            best_val_map = val_map
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_map.pt'))
            print("saved model new best mAP")

        # plot loss and metrics
        plot_loss(train_losses, val_losses, val_map_all, save_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_ROOT', type=str, default=None)
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=None)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=None)
    parser.add_argument('--CONFIG', type=str)
    parser.add_argument('--RUN_ID', type=str, default=None)
    parser.add_argument('--INFERENCE', action=argparse.BooleanOptionalAction, default=False)
    config = parser.parse_args()
    cmd_config = vars(config)

    # load model and training configs
    with open('config/' + cmd_config['CONFIG'] + '.yaml') as f:
        yaml_config = yaml.load(f, yaml.FullLoader)

    config = yaml_config
    for k, v in cmd_config.items():
        if v is not None:
            config[k] = v

    print('config: ', config)

    main(config)