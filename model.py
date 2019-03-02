import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, Resize

from os import listdir, makedirs, remove
from os.path import exists, join, basename
from PIL import Image, ImageFilter
from six.moves import urllib
import tarfile
import argparse
from math import log10



class SRCNN(nn.Module):
	def __init__(self, upscale_factor=3, learning_rate=0.0001):
		super(SRCNN, self).__init__()

		# Patch extraction and representation
		self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
		self.relu1 = nn.ReLU()

		# Non linear mapping
		self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
		self.relu2 = nn.ReLU()

		# Reconstruction
		self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)


		# Init values
		self.epoch = 1;
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		self.criterion = nn.MSELoss()
		self.training_data_loader = None


	def forward(self, x):
		out = self.conv1(x)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.relu2(out)
		out = self.conv3(out)

		return out

	def train(self, training_data_loader=None):
		if (training_data_loader is None):
			training_data_loader = self.training_data_loader
		self.training_data_loader = training_data_loader

		epoch_loss = 0
		avg_psnr = 0
		for iteration, batch in enumerate(training_data_loader, 1):
			input, target = Variable(batch[0]), Variable(batch[1])
			if self.use_cuda:
				input = input.cuda()
				target = target.cuda()

			self.optimizer.zero_grad()
			out = self.forward(input)

			loss = self.criterion(out, target)
			epoch_loss += loss.data
			loss.backward()
			self.optimizer.step()

			psnr = 10 * log10(1 / loss.data)
			avg_psnr += psnr

			# print("===> Epoch[{}]({}/{}): Loss: {:.4f} PSNR: {:.4f} dB".format(self.epoch, iteration, len(training_data_loader), loss.data, psnr))

		print("===> Epoch {} Complete: Avg. Loss: {:.4f} Avg. PSNR: {:.4f} dB".format(self.epoch, epoch_loss / len(training_data_loader), avg_psnr / len(testing_data_loader)))
		self.epoch += 1


	def use_cuda(self):
		self.cuda()
		self.criterion = self.criterion.cuda()
		self.use_cuda = True


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
	img = Image.open(filepath)
	# .convert('YCbCr')

	return img


class DatasetFromFolder(data.Dataset):
	def __init__(self, image_dir, input_transform=None, target_transform=None):
		super(DatasetFromFolder, self).__init__()
		self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		input = load_img(self.image_filenames[index])
		target = input.copy()
		if self.input_transform:
			# print(input.mode)
			input = input.filter(ImageFilter.GaussianBlur(2))

			input = self.input_transform(input)
		if self.target_transform:
			target = self.target_transform(target)

		return input, target

	def __len__(self):
		return len(self.image_filenames)


def download_bsd500(dest="dataset"):
	output_image_dir = join(dest, "BSR/BSDS500/data/images")

	if not exists(output_image_dir):
		makedirs(dest)
		url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
		print("downloading url ", url)

		data = urllib.request.urlopen(url)

		file_path = join(dest, basename(url))
		with open(file_path, 'wb') as f:
			f.write(data.read())

		print("Extracting data")
		with tarfile.open(file_path) as tar:
			for item in tar:
				tar.extract(item, dest)

		remove(file_path)

	return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
	return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
	return Compose([
		CenterCrop(crop_size),
		Resize(crop_size // upscale_factor),
		Resize(crop_size, interpolation=Image.BICUBIC),
		ToTensor(),
	])


def target_transform(crop_size):
	return Compose([
		CenterCrop(crop_size),
		ToTensor(),
	])


def get_training_set(upscale_factor):
	root_dir = download_bsd500()

	train_dir = join(root_dir, "train")
	crop_size = calculate_valid_crop_size(256, upscale_factor)

	return DatasetFromFolder(train_dir,
							 input_transform=input_transform(crop_size, upscale_factor),
							 target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
	root_dir = download_bsd500()
	test_dir = join(root_dir, "test")
	crop_size = calculate_valid_crop_size(256, upscale_factor)

	return DatasetFromFolder(test_dir,
							 input_transform=input_transform(crop_size, upscale_factor),
							 target_transform=target_transform(crop_size))

if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	if (use_cuda):
		print("Using cuda")

	train_set = get_training_set(3)
	test_set = get_test_set(3)
	training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=4, shuffle=False)

	srcnn = SRCNN()

	if (use_cuda):
		srcnn.use_cuda()

	for epoch in range(150):
		srcnn.train(training_data_loader)