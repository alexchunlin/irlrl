import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transforms import TanhTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import ipdb

from utils import numpify

epsilon = 1e-6

class DeterministicPolicy(nn.Module):
	def __init__(self,state_dim,a_dim,h_dim,action_space=None):

		super().__init__()		
		self.net = nn.Sequential(nn.Linear(state_dim,h_dim),
							nn.Tanh(),
							nn.Linear(h_dim,h_dim),
							nn.Tanh(),
							nn.Linear(h_dim,a_dim),
							nn.Tanh())
				
		self.a_dim = a_dim

		if action_space is None:
			self.action_scale = torch.tensor(1.)
			self.action_bias = torch.tensor(0.)
		else:
			self.action_scale = torch.FloatTensor(
				(action_space.high - action_space.low) / 2.)
			self.action_bias = torch.FloatTensor(
				(action_space.high + action_space.low) / 2.)

	def forward(self,state):

		x = self.net(state)
		a = self.action_scale*x + self.action_bias
		
		return a
	
	def sample(self,state,eps=None):

		a = self.forward(state)
		log_prob = torch.zeros_like(a).sum(-1)

		return a,log_prob,a



	def reset_state(self,batch_size=None):
		# this is not a recurrent policy so we don't have any state to reset
		pass

	def np_policy(self,state):

		with torch.no_grad():
			state = torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float)
			action = self.forward(state)
		action = action.cpu().detach().numpy()

		return action

	def to(self, device):
		self.action_scale = self.action_scale.to(device)
		self.action_bias = self.action_bias.to(device)
		return super().to(device)


class TanhGaussianPolicy(nn.Module):

	def __init__(self,state_dim,a_dim,h_dim,action_space=None,device=torch.device('cuda:0'),std_scale=0.1,max_std=False):
		'''
		std_scale: parameter that network std is multiplied by (found it was more stable for MBRL to set to .1)
		max_std: if True, caps the maximum std to std_scale.
		'''

		super().__init__()
		
		self.layers = nn.Sequential(nn.Linear(state_dim,h_dim),
							nn.Tanh(),
							nn.Linear(h_dim,h_dim),
							nn.Tanh())
		self.mean_layer = nn.Sequential(nn.Linear(h_dim,a_dim))
		if not max_std:
			self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Softplus())
		else:
			self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Sigmoid())
		self.std_scale = std_scale

		self.a_dim = a_dim
		self.device = device

		# action rescaling
		if action_space is None:
			self.action_scale = torch.tensor(1.)
			self.action_bias = torch.tensor(0.)
		else:
			self.action_scale = torch.FloatTensor(
				(action_space.high - action_space.low) / 2.)
			self.action_bias = torch.FloatTensor(
				(action_space.high + action_space.low) / 2.)

	def forward(self,state):

		x = self.layers(state)
		mean = self.mean_layer(x)
		sig  = self.std_scale*self.sig_layer(x)

		return mean,sig

	def sample(self,state,eps=None):
		mean,std = self.forward(state)
		normal = Normal(mean,std)

		if eps is None:
			x_t = normal.rsample()
		else:
			x_t = mean + std*eps
		y_t = torch.tanh(x_t)
		action = y_t *self.action_scale + self.action_bias
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
		log_prob = log_prob.sum(-1, keepdim=True)
		mean = torch.tanh(mean) * self.action_scale + self.action_bias
		return action, log_prob, mean

	def np_policy(self,state):
		with torch.no_grad():
			state = torch.tensor(state,device=self.device,dtype=torch.float32)
			action,_,_ = self.sample(state)
			action = numpify(action)
		return action

	def reset_state(self,batch_size=None):
		# this is not a recurrent policy so we don't have any state to reset
		pass

	def to(self, device):
		self.action_scale = self.action_scale.to(device)
		self.action_bias = self.action_bias.to(device)
		return super().to(device)


class GaussianPolicy(nn.Module):

	def __init__(self,state_dim,a_dim,h_dim,action_space=None,device=torch.device('cuda:0'),std_scale=0.1,max_std=False):

		super().__init__()
		
		self.layers = nn.Sequential(nn.Linear(state_dim,h_dim),
							nn.Tanh(),
							nn.Linear(h_dim,h_dim),
							nn.Tanh())
		self.mean_layer = nn.Sequential(nn.Linear(h_dim,a_dim))
		if not max_std:
			self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Softplus())
		else:
			self.sig_layer = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Sigmoid())
		self.std_scale = std_scale

		self.a_dim = a_dim
		self.device = device

		# action rescaling
		if action_space is None:
			self.action_scale = torch.tensor(1.)
			self.action_bias = torch.tensor(0.)
		else:
			self.action_scale = torch.FloatTensor(
				(action_space.high - action_space.low) / 2.)
			self.action_bias = torch.FloatTensor(
				(action_space.high + action_space.low) / 2.)

	def forward(self,state):

		x = self.layers(state)
		mean = self.mean_layer(x)
		sig  = self.std_scale*self.sig_layer(x)

		return mean,sig

	def sample(self,state,eps=None):
		mean,std = self.forward(state)
		normal = Normal(mean,std)

		if eps is None:
			action = normal.rsample()
		else:
			action = mean + std*eps
		y_t = torch.tanh(x_t)
		action = y_t *self.action_scale + self.action_bias
		log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
		
		return action, log_prob, mean

	def np_policy(self,state):
		with torch.no_grad():
			state = torch.tensor(state,device=self.device,dtype=torch.float32)
			action,_,_ = self.sample(state)
			action = numpify(action)
		return action

	def reset_state(self,batch_size=None):
		# this is not a recurrent policy so we don't have any state to reset
		pass

	def to(self, device):
		self.action_scale = self.action_scale.to(device)
		self.action_bias = self.action_bias.to(device)
		return super().to(device)