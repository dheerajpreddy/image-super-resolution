import os
from PIL import Image

def saveSubImages(img, dest_dir, image_name, img_size=32, stride=14):
	h, w = img.size

	images = []

	for i in range(0, h, stride):
		if (i + img_size >= h):
			break
		for j in range(0, w, stride):
			if (j + img_size >= w):
				break
			images.append(img.crop((i, j, i+img_size, j+img_size)))

	for i, image in enumerate(images):
		image.save(dest_dir + image_name.replace(".", "-{}.".format(i+1)))

	print(image_name)


if __name__ == '__main__':
	filenames = os.listdir("./dataset/T91/")
	filenames = [x for x in filenames if ".png" in x]

	for name in filenames:
		img = Image.open("./dataset/T91/" + name)
		saveSubImages(img, "./dataset/T91-subimages/", name)
