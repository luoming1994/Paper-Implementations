from __future__ import print_function
import argparse
import numpy as np
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
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
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
train_dataset = dset.MNIST(root = '../data/',
					train = True,
                    transform = transforms.ToTensor(),
					#transform=transforms.Compose([transforms.Scale(opt.imageSize),transforms.ToTensor()]),
                    download = True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                     batch_size = opt.batchSize,
                                     shuffle = True)
									 
test_dataset = dset.MNIST(root = '../data/',
                              train = False,
                              transform = transforms.ToTensor(),
                              download = True)									 
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = opt.batchSize,
                                           shuffle = False)
									 
#other args
cudnn.benchmark = True
nc = 1
net = leNet(nc,opt.ndf)
if(opt.cuda):
    net.cuda()

optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)	
criterion = nn.CrossEntropyLoss()
net.train()
with open(r'sample/out.txt','a') as f:
	for epoch in range(opt.niter):
		loss_list= []
		acc_list = []
		for indx,(images,labels) in enumerate(train_loader):
			net.zero_grad()
			#images.data.resize_(images_.size()).copy_(images_)
			#label.data.resize_(label_.size()).copy_(label_)
			if(opt.cuda):
				images = images.cuda()
				labels = labels.cuda()
			images = Variable(images)
			labels = Variable(labels)
		
			out = net(images)
			err = criterion(out,labels)
			err.backward()
			optimizer.step()
			# calculate acc
			pre = out.cpu().data.numpy().argmax(axis=1)
			labels_np = labels.cpu().data.numpy()
			acc = np.mean(pre ==labels_np)
			# write loss and acc into file 
			write_str = '%f %f\n'%(err.data[0],acc)
			f.write(write_str)
			loss_list.append(err.data[0])
			acc_list.append(acc)
		#print epoch mean loss and acc			
		print('[%d/%d]Loss: %.4f	Acc: %.4f'
                  % (epoch, opt.niter,
                    sum(loss_list)/len(loss_list),sum(acc_list)/len(acc_list)))

net.eval()
loss_list= []
acc_list = []
for indx,(images,labels) in enumerate(test_loader):
	if(opt.cuda):
		images = images.cuda()
		labels = labels.cuda()
	images = Variable(images)
	labels = Variable(labels)

    # forward
	out = net(images)
	err = criterion(out,labels)
	
	pre = out.cpu().data.numpy().argmax(axis=1)
	labels_np = labels.cpu().data.numpy()
	acc = np.mean(pre ==labels_np)
	acc_list.append(acc) 
	loss_list.append(err.data[0])
	#print('[%d/%d]Acc:%.4f' %(indx, len(test_loader),acc) )
# logging
print('[eval]Loss: %.4f	Acc: %.4f. '%(sum(loss_list)/len(loss_list),sum(acc_list)/len(acc_list)) )
	
torch.save(net.state_dict(), '%s/net.pth' % (opt.outf))	
loss = np.loadtxt('sample/out.txt')
