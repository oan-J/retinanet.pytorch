import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type,csv.')
	parser.add_argument('--csv_classes', help='Path to file containing class list')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations')

	parser.add_argument('--model', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	if parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	# no cuda
	retinanet = torch.load(parser.model, map_location=torch.device('cpu'))
	use_gpu = False

	# cuda
	# retinanet = torch.load(parser.model)
	# use_gpu = True

	if use_gpu:
		if torch.cuda.is_available():
			retinanet = retinanet.cuda()

	if torch.cuda.is_available():
		# retinanet = torch.nn.DataParallel(retinanet).cuda()
		device = torch.device('cuda')
		retinanet = retinanet.to(device)
		if torch.cuda.device_count() > 1:
			retinanet = torch.nn.DataParallel(retinanet)
	else:
		device = torch.device('cpu')
		retinanet = retinanet.module.to(device)

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):
		print(idx)
		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			else:
				scores, classification, transformed_anchors = retinanet(data['img'].float())
			print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores.cpu()>0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(label_name)

			cv2.imshow('img', img)
			cv2.waitKey(0)	# 接收键盘上的值



if __name__ == '__main__':
 main()