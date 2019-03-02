import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, Resize
import torchvision.transforms as transforms

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
		self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
		self.relu1 = nn.ReLU()

		# Non linear mapping
		self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
		self.relu2 = nn.ReLU()

		# Reconstruction
		self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)


		# Init values
		self.epoch = 1;
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		self.criterion = nn.MSELoss()
		self.training_data_loader = None
		self.test_data_loader = None


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
		total = 0
		saved = False
		for iteration, batch in enumerate(training_data_loader, 1):
			input, target = Variable(batch[0]), Variable(batch[1])
			if self.use_cuda:
				input = input.cuda()
				target = target.cuda()

			self.optimizer.zero_grad()
			out = self.forward(input)

			if saved == False:
				saved = True
				torch.save(input, "input.tensor")
				torch.save(out, "out.tensor")
				torch.save(target, "target.tensor")

			loss = self.criterion(out[:, :, 12:-12, 12:-12], target[:, :, 12:-12, 12:-12])
			epoch_loss += loss.data
			loss.backward()
			self.optimizer.step()

			psnr = 10 * log10(1 / loss.data)
			avg_psnr += psnr
			total += 1

			print("===> Epoch[{}]({}/{}): Loss: {:.4f} PSNR: {:.4f} dB".format(self.epoch, iteration, len(training_data_loader), loss.data, psnr))
			# print(len(training_data_loader))

		print("===> Epoch {} Complete: Avg. Loss: {:.4f} Avg. PSNR: {:.4f} dB".format(self.epoch, epoch_loss / total, avg_psnr / total))
		self.epoch += 1


	def test(self, test_data_loader=None):
		if (test_data_loader is None):
			test_data_loader = self.test_data_loader
		self.test_data_loader = test_data_loader

		tt = transforms.ToPILImage()

		epoch_loss = 0
		avg_psnr = 0
		bicubic_avg_psnr = 0
		total = 0
		imgIdx = 1
		for iteration, batch in enumerate(test_data_loader, 1):
			input, target = Variable(batch[0]), Variable(batch[1])
			if self.use_cuda:
				input = input.cuda()
				target = target.cuda()
			out = self.forward(input)

			for i, currImg in enumerate(out):
				temp = input[i].cpu()
				print(temp.shape)
				inputImg = tt(temp)
				print(inputImg.size)
				inputImg.save("./results/{}-input.jpg".format(imgIdx))
				outImg = tt(out[i].cpu())
				outImg.save("./results/{}-out.jpg".format(imgIdx))
				targetImg = tt(target[i].cpu())
				targetImg.save("./results/{}-target.jpg".format(imgIdx))
				imgIdx += 1

			loss = self.criterion(out, target)
			bicubic_loss = self.criterion(input, target)
			epoch_loss += loss.data
			bicubic_epoch_loss = self.criterion(input, target)

			psnr = 10 * log10(1 / loss.data)
			bicubic_psnr = 10 * log10(1 / bicubic_loss.data)
			avg_psnr += psnr
			bicubic_avg_psnr += bicubic_psnr
			total += 1

			# print("===> Test[{}]({}/{}): Loss: {:.4f} PSNR: {:.4f} dB".format(self.epoch, iteration, len(test_data_loader), loss.data, psnr))
			# print(len(test_data_loader))

		print("===> Test {} Complete: Avg. Loss: {:.4f} Avg. PSNR: {:.4f} dB".format(self.epoch, epoch_loss / total, avg_psnr / total))
		print("===> Bicubic Test {} Complete: Avg. Loss: {:.4f} Avg. PSNR: {:.4f} dB".format(self.epoch, bicubic_epoch_loss / total, bicubic_avg_psnr / total))


	def set_cuda(self):
		self.cuda()
		self.criterion = self.criterion.cuda()
		self.use_cuda = True

	def save_checkpoint(self):
		model_out_path = "model_epoch_{}.pth".format(self.epoch - 1)
		torch.save(self.state_dict(), model_out_path)
		print("Checkpoint saved to {}".format(model_out_path))



def is_image_file(filename):
	return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
	img = Image.open(filepath)
	# .convert('YCbCr')
	# y, cb, cr = img.split()

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


def input_transform(crop_size=None, upscale_factor=1.0):
	if crop_size is None:
		return Compose([
			ToTensor(),
		])

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
	train_dir = "./dataset/SR_training_datasets/BSDS200"
	crop_size = calculate_valid_crop_size(256, upscale_factor)

	return DatasetFromFolder(train_dir,
							 input_transform=input_transform(crop_size, upscale_factor),
							 target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
	root_dir = download_bsd500()
	test_dir = join(root_dir, "test")
	test_dir = "./dataset/Set5"
	crop_size = calculate_valid_crop_size(32, upscale_factor)

	return DatasetFromFolder(test_dir,
							 input_transform=input_transform(upscale_factor=upscale_factor),
							 target_transform=input_transform(upscale_factor=upscale_factor))

if __name__ == '__main__':


	use_cuda = torch.cuda.is_available()
	if (use_cuda):
		print("Using cuda")

	torch.manual_seed(123)
	if use_cuda:
	    torch.cuda.manual_seed(123)

	train_set = get_training_set(3)
	test_set = get_test_set(3)
	training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

	# srcnn = torch.load("./model_epoch_1.pth")
	# print(dir(srcnn))
	srcnn = SRCNN()

	if (use_cuda):
		srcnn.set_cuda()

	for epoch in range(150):
		srcnn.train(training_data_loader)
		srcnn.save_checkpoint()
		srcnn.test(testing_data_loader)