import argparse
import collections
import os
import numpy as np
import datetime
import pickle
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
# from writeTB import write_tensorboard
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
writer = None


def main(args=None):
    global writer
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, csv')
    parser.add_argument('--csv_train', help='Path to file containing training annotations')
    parser.add_argument('--csv_classes', help='Path to file containing class list')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=15)
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--save-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    parser = parser.parse_args(args)

    if parser.tensorboard:
        writer = SummaryWriter()
    if parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('parser.csv_train is None')

        if parser.csv_classes is None:
            raise ValueError('parser.csv_classes')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        if parser.csv_val is None:
            dataset_val = None
            print('parser.csv_val is None')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv ), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if parser.tensorboard and writer is not None:
                    writer.add_scalar('Loss/train', float(loss), epoch_num)
                    writer.add_scalar('Train/Classification_loss', float(classification_loss),
                                      epoch_num * len(dataloader_train) + iter_num)
                    writer.add_scalar('Train/Regression_loss', float(regression_loss),
                                      epoch_num * len(dataloader_train) + iter_num)
                    writer.add_scalar('Train/Total_loss', float(loss), epoch_num * len(dataloader_train) + iter_num)

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)
            # if parser.tensorboard and writer is not None:
            #     writer.add_scalar('mAP', mAP['mAP'], epoch_num)
        else:
            raise ValueError('parser.dataset == csv or parser.csv_val is  None')

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()
    test_loss_hist = []  # 用于保存测试集损失的列表

    # for iter_num, data in enumerate(dataloader_val):
    #     with torch.no_grad():
    #         if torch.cuda.is_available():
    #             classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
    #         else:
    #             classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

    #         classification_loss = classification_loss.mean()
    #         regression_loss = regression_loss.mean()

    #         loss = classification_loss + regression_loss

    #         test_loss_hist.append(float(loss))  # 保存损失值

    #         print('Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Test loss: {:1.5f}'.format(
    #             iter_num, float(classification_loss), float(regression_loss), float(loss)))

    #         del classification_loss
    #         del regression_loss

    # mean_test_loss = np.mean(test_loss_hist)  # 计算测试集损失的平均值

    # if parser.tensorboard and writer is not None:
    #     writer.add_scalar('Val Loss/test', mean_test_loss, epoch_num * len(dataloader_train) + iter_num)  # 将测试集损失写入TensorBoard

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
    writer.close()