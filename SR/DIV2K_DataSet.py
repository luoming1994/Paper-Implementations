from os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

 
def is_image_file(filename):
	"""
	a file is a image? via extension
	"""
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath,upscale=None):
	"""
	load a image and convert to YCbCr,return Y channel
	"""
    img = Image.open(filepath)
	if upscale is not None:
		W,H = img.size
		img = img.resize((W*upscale, H*upscale),Image.BICUBIC))
	img_YCbCr = img.convert('YCbCr')
    y, _, _ = img_YCbCr.split()
    return y

def crop_img(img,scale_num):
	"""
	crop image with integer multiple of scale_num
	"""
	if isinstance(img, Image.Image):
		W,H = img.size
		W_crop , H_crop = W//scale_num , H//scale_num
		img_crop = img.crop((0,0,W_crop,H_crop))  

	return img_crop
	

def cut_img(img,crop_size = 64):
	"""
	cut image(torch.Tensor) into normal size with size(crop_size * crop_size)
	"""
	if isinstance(img, torch.Tensor):
		img_size = img.size()
		assert len(img_size)==3 
		if img_size[-1]>=64 and img_size[-2]>=64:
			H,W = img_size[-2]//crop_size,img_size[-2]//crop_size
			chnnl = img_size[0]   # img channel
			crop_img  = torch.Tensor(H*W*chnnl,crop_size,crop_size)
			for h in range(0,H,1):
				for w in range(0,W,1):
					indx = h*W+w
					crop_img[indx*chnnl:(indx+1)*chnnl,:,:] = img[:,h*64:(h+1)*64,w*64:(w+1)*64]
	
    return crop_img

class DIV2K_DataSet(data.Dataset):
	"""
	super resolution image dataset
	data_dir:  images file dir
	scale_list:
	"""
	def __init__(self, data_dir,scale_list=[2],):
		super(DIV2K_DataSet, self).__init__()
		# HR images pathname list
        self.paths_HR = [os.path.join(data_dir, '%04d.png'%x) for x in range(1,801,1)]
		#[2,3,4]^scale_list;intersection of two list
		self.scale_list = [val for val in [2,3,4] if val in scale_list]	
		# LR images pathname list , everyone element is a list which contains LR image pathnames
		self.paths_LR = []	
		for upscale in self.scale_list:
			paths_LR = [os.path.join(data_dir,
								'DIV2K_train_LR_bicubic', 'X%d'%upscale,'%04dx%d.png'%(x,upscale)) 
								for x in range(1,801,1)]
			self.paths_LR.append(paths_LR)

		# load all image
		self.data = torch.Tensor()
		self.label = torch.Tensor()
		for idx,upscale in enumerate(self.scale_list):	# upscale 2,3,4
			for filename in self.paths_HR:
				img_hr = load_img(filename)
				img_hr = transforms.ToTensor()(img_hr)
				img_hr_crop = cut_img(img_hr)
				self.data = torch.cat((self.data,img_hr_crop),dim=0)
			for filename in self.paths_LR[idx]:		# upscale 
				img_lr = load_img(filename,upscale=upscale)
				img_lr = transforms.ToTensor()(image_lr)
				img_lr_crop = cut_img(image_lr)
				self.label = torch.cat((self.label,img_lr_crop),dim=0)
		
		
		

    def __getitem__(self, index):
        
		#input = load_img(self.image_filenames[index])
        #target = input.copy()
        #if self.input_transform:
		#    # 1*H*W
        #    input = self.input_transform(input)
        #if self.target_transform:
        #    target = self.target_transform(target)
		# H*W 2 dim
		input = self.data[indx]
		target= self.label[indx]
		
        return input, target

    def __len__(self):
        #return len(self.image_filenames)
		return self.data.size()[0]
		