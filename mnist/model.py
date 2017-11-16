import torch
import torch.nn as nn

class leNet(nn.Module):
	def __init__(self,nc,ndf):
		"""
		nc: input channel
		ndf:middle channel
		"""
		super(leNet,self).__init__()
		# 32*32
		self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
								nn.BatchNorm2d(ndf),
								nn.ReLU(inplace=True),
								nn.MaxPool2d(kernel_size=2,stride=2))
		# 16*16
		self.conv2 = nn.Sequential(nn.Conv2d(ndf,2*ndf,kernel_size=3,stride=1,padding=1),
								nn.BatchNorm2d(2*ndf),
								nn.ReLU(inplace=True),
								nn.MaxPool2d(kernel_size=2,stride=2))	
		# 8*8
		self.conv3 = nn.Sequential(nn.Conv2d(2*ndf,4*ndf,kernel_size=3,stride=1,padding=1),
								nn.BatchNorm2d(4*ndf),
								nn.ReLU(inplace=True),
								nn.MaxPool2d(kernel_size=2,stride=2))
		# 4*4
		self.fc = nn.Linear(4*ndf*4*4,10)
		self.ndf = ndf
	
	def forward(self,x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = out.view(out.size(0),-1)
		out = self.fc(out)
		return out