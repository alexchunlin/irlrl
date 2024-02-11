import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class ValueNet(nn.Module):
	def __init__(self, num_inputs, hidden_dim):
		super(ValueNet, self).__init__()

		self.linear1 = nn.Linear(num_inputs, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)

	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

class QNetwork(nn.Module):
	def __init__(self, state_dim, a_dim, h_dim, layer_norm=False):
		super().__init__()
		if layer_norm:
			self.layers = nn.Sequential(nn.Linear(state_dim+a_dim,h_dim),nn.LayerNorm(h_dim),nn.ReLU(),
										nn.Linear(h_dim,h_dim),          nn.LayerNorm(h_dim),nn.ReLU(),
										nn.Linear(h_dim,1))

		else:
			self.layers = nn.Sequential(nn.Linear(state_dim+a_dim,h_dim),nn.ReLU(),
										nn.Linear(h_dim,h_dim),          nn.ReLU(),
										nn.Linear(h_dim,1))
	def forward(self, state, action):
		# ipdb.set_trace()
		try:
			state_action = torch.cat([state,action],dim=-1)
		except:
			ipdb.set_trace()
		return self.layers(state_action)


class EnsembleQNetwork(nn.Module):
	'''
	This is going to be a REDQ-style critic (ensemble of Q functions)
	it needs to be able to call a random subset of critics
	it also needs to be able to call all critics
	'''
	def __init__(self, state_dim, a_dim, h_dim, n_members, layer_norm=False):
		super().__init__()
		self.Q_nets = nn.ModuleList([QNetwork(state_dim, a_dim, h_dim, layer_norm=layer_norm) for i in range(n_members)])
		self.n_members = n_members
		self.member_inds = np.arange(self.n_members)
		
	def forward(self,state,action,subset_size=None):

		if subset_size is None:
			inds = self.member_inds # everything
		else:
			inds = np.random.choice(self.member_inds,subset_size,replace=False) # random subset of size subset_size

		return torch.stack([self.Q_nets[i](state,action) for i in inds])