from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model import leNet



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='sample/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')


opt = parser.parse_args()
print(opt)

# output dir
try:
    os.makedirs(opt.outf)
except OSError:
    pass
# random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
## cuda
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
###############   DATASET   ##################
dataset = dset.MNIST(root = '../data/',
                    transform=transforms.Compose([
                    transforms.Scale(opt.imageSize),
                    transforms.ToTensor(),
                    ]),
                    download = True)
loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.batchSize,
                                     shuffle = True)
#other args
cudnn.benchmark = True
nc = 1
net = leNet(nc,opt.ndf)
if(opt.cuda):
    net.cuda()

#images = torch.FloatTensor(opt.batchSize,nc,opt.imageSize,opt.imageSize)
#label = torch.FloatTensor(opt.batchSize,nc,1,1)
#images = Variable(images)
#label = Variable(label)
#if(opt.cuda):
#    images = images.cuda()
#    label = label.cuda()


optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)	
criterion = nn.CrossEntropyLoss()
for epoch in range(opt.niter+1):
	for indx,(images,label) in enumerate(loader):
		net.zero_grad()
		#images.data.resize_(images_.size()).copy_(images_)
		#label.data.resize_(label_.size()).copy_(label_)
		if(opt.cuda):
			images = images.cuda()
			label = label.cuda()
		images = Variable(images)
		label = Variable(label)
		
		out = net(images)
		err = criterion(out,label)
		err.backward()
		optimizer.step()
	
		print('[%d/%d][%d/%d] Loss: %.4f'
                  % (epoch, opt.niter, indx, len(loader),
                     err.data[0]))
	
torch.save(net.state_dict(), '%s/net.pth' % (opt.outf))	