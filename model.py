import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, Resize
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

from os import listdir, makedirs, remove
from os.path import exists, join, basename
from PIL import Image, ImageFilter
import argparse
from math import log10

vgg16 = models.vgg16(pretrained=True)

class SRCNN(nn.Module):
	def __init__(self, upscale_factor=3, learning_rate=0.0001):
		super(SRCNN, self).__init__()

		# Patch extraction and representation
		self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)

		# Non linear mapping
		self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)

		# Reconstruction
		self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)


		# Init values
		self.epoch = 1;
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		self.criterion = nn.MSELoss()
		self.training_data_loader = None
		self.test_data_loader = None


	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.relu(self.conv2(out))
		out = self.conv3(out)

		return out


	def train(self, training_data_loader=None):
		if (training_data_loader is None):
			training_data_loader = self.training_data_loader
		self.training_data_loader = training_data_loader

		tt = transforms.ToPILImage()

		epoch_loss = 0
		avg_psnr = 0
		total = 0
		for iteration, batch in enumerate(training_data_loader, 1):
			(input, cb, cr), target = batch[0], Variable(batch[1])
			input = Variable(input)
			if self.use_cuda:
				input = input.cuda()
				target = target.cuda()

			self.optimizer.zero_grad()
			out = self.forward(input)

			out = torch.cat((out.cuda(), cb.cuda(), cr.cuda()), dim=1)
			target = torch.cat((target, cb.cuda(), cr.cuda()), dim=1)

			loss = self.criterion(vgg16(out), vgg16(target))
			epoch_loss += loss.data
			loss.backward()
			self.optimizer.step()

			psnr = 10 * log10(1 / loss.data)
			avg_psnr += psnr
			total += 1

			print("===> Epoch[{}]({}/{}): Loss: {:.4f} PSNR: {:.4f} dB".format(self.epoch, iteration, len(training_data_loader), loss.data, psnr))
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
			(input, cb, cr), target = batch[0], Variable(batch[1])
			input = Variable(input)

			if self.use_cuda:
				input = input.cuda()
				target = target.cuda()
			out = self.forward(input)

			for i, currImg in enumerate(out):
				CB = tt(cb[i]).convert('L')
				CR = tt(cr[i]).convert('L')
				inputImg = tt(input[i].cpu()).convert('L')
				inputImg = Image.merge('YCbCr', (inputImg, CB, CR))
				inputImg.save("./results/{}-input.jpg".format(imgIdx))

				outImg = tt(out[i].clamp_(0, 1).cpu()).convert('L')
				outImg = Image.merge('YCbCr', (outImg, CB, CR))
				outImg.save("./results/{}-out.jpg".format(imgIdx))

				targetImg = tt(target[i].cpu()).convert('L')
				targetImg = Image.merge('YCbCr', (targetImg, CB, CR))
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

		print("===> Test {} Complete: Avg. Loss: {:.4f} Avg. PSNR: {:.4f} dB".format(self.epoch, epoch_loss / total, avg_psnr / total))
		print("===> Bicubic Test {} Complete: Avg. Loss: {:.4f} Avg. PSNR: {:.4f} dB".format(self.epoch, bicubic_epoch_loss / total, bicubic_avg_psnr / total))


	def set_cuda(self):
		self.cuda()
		self.criterion = self.criterion.cuda()
		self.use_cuda = True
		vgg16.cuda()

	def save_checkpoint(self):
		model_out_path = "model_epoch_{}.pth".format(self.epoch - 1)
		torch.save(self.state_dict(), model_out_path)
		print("Checkpoint saved to {}".format(model_out_path))



def is_image_file(filename):
	return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
	img = Image.open(filepath).convert('YCbCr')
	return img


class DatasetFromFolder(data.Dataset):
	def __init__(self, image_dir, input_transform=None, target_transform=None):
		super(DatasetFromFolder, self).__init__()
		self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		input, cb, cr = load_img(self.image_filenames[index]).split()

		target = input.copy()
		if self.input_transform:
			input = input.filter(ImageFilter.GaussianBlur(2))
			input = self.input_transform(input)

		if self.target_transform:
			target = self.target_transform(target)
			cb = self.target_transform(cb)
			cr = self.target_transform(cr)

		return (input, cb, cr), target

	def __len__(self):
		return len(self.image_filenames)


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
	train_dir = "./dataset/T91-subimages"
	crop_size = calculate_valid_crop_size(32, upscale_factor)

	return DatasetFromFolder(train_dir,
							 input_transform=input_transform(crop_size, upscale_factor),
							 target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
	test_dir = "./dataset/Set5"
	crop_size = calculate_valid_crop_size(32, upscale_factor)

	return DatasetFromFolder(test_dir,
							 input_transform=input_transform(upscale_factor=upscale_factor),
							 target_transform=input_transform(upscale_factor=upscale_factor))

if __name__ == '__main__':
	# print(vgg16.children())
	vgg16 = torch.nn.Sequential(*list(vgg16.children())[0][:16])
	# print(vgg16)

	use_cuda = torch.cuda.is_available()
	if (use_cuda):
		print("Using cuda")

	torch.manual_seed(123)
	if use_cuda:
	    torch.cuda.manual_seed(123)

	train_set = get_training_set(3)
	test_set = get_test_set(3)
	training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, shuffle=True)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)


	# srcnn = torch.load("./model_epoch_1.pth")
	# print(dir(srcnn))
	srcnn = SRCNN()

	if (use_cuda):
		srcnn.set_cuda()

	print("TRAINING")
	for epoch in range(150):
		srcnn.train(training_data_loader)
		srcnn.save_checkpoint()
		srcnn.test(testing_data_loader)