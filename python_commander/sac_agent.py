import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from critic import EnsembleQNetwork
from policy import TanhGaussianPolicy, GaussianPolicy, DeterministicPolicy
import ipdb


def build_policy(state_dim,a_dim,args,action_space=None,device=torch.device('cuda:0')):

	if args.policy_type == 'deterministic':
		raise NotImplementedError
		policy = DeterministicPolicy(state_dim,a_dim,args.h_dim,action_space=action_space).to(device)
	elif args.policy_type == 'tanh_gaussian':
		policy = TanhGaussianPolicy(state_dim,a_dim,args.h_dim,action_space=action_space,std_scale=args.mf_std_scale).to(device)
	elif args.policy_type == 'gaussian':
		policy = GaussianPolicy(state_dim,a_dim,args.h_dim,action_space=action_space,std_scale=args.mf_std_scale).to(device)


	return policy

class SAC(object):
	def __init__(self,state_dim,action_space,args,automatic_entropy_tuning=False):

		self.gamma = args.gamma
		self.tau = args.tau
		self.alpha = args.alpha_mf
		self.subset_size = args.q_subset_size # "M" parameter from REDQ https://arxiv.org/pdf/2101.05982.pdf
		self.args = args
		self.device = torch.device("cpu" if args.cpu else "cuda")
		self.updates = 0
		self.action_space = action_space

		self.policy_type = args.policy_type
		self.target_update_interval = args.target_update_interval
		self.automatic_entropy_tuning = args.automatic_entropy_tuning or automatic_entropy_tuning

		self.policy = build_policy(state_dim,action_space.shape[0],args,action_space)
		self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


		
											# state_dim, a_dim,                      h_dim,                        n_members,layer_norm=False
		self.critic        = EnsembleQNetwork(state_dim, action_space.shape[0], args.h_dim,n_members=args.q_ensemble_members,layer_norm=args.q_layer_norm).to(device=self.device)
		self.critic_target = EnsembleQNetwork(state_dim, action_space.shape[0], args.h_dim,n_members=args.q_ensemble_members,layer_norm=args.q_layer_norm).to(device=self.device)
		self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
		hard_update(self.critic_target, self.critic)

		
		
		self.QLoss = nn.MSELoss()
		
		if self.automatic_entropy_tuning:

			self.target_entropy = -args.target_entropy*torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
			self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
			self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=4e-3)
		

	def select_action(self, state, evaluate=False):
		state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
		if evaluate is False:
			action, _, _ = self.policy.sample(state)
		else:
			_, _, action = self.policy.sample(state)
		return action.detach().cpu().numpy()[0]

	def update_parameters(self, replay_buffer, batch_size):
		state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size=batch_size)

		state_batch      = torch.tensor(state_batch,     dtype=torch.float32,device=self.device)
		next_state_batch = torch.tensor(next_state_batch,dtype=torch.float32,device=self.device)
		action_batch     = torch.tensor(action_batch,    dtype=torch.float32,device=self.device)
		reward_batch     = torch.tensor(reward_batch,    dtype=torch.float32,device=self.device).unsqueeze(1)
		mask_batch       = torch.tensor(mask_batch,      dtype=torch.float32,device=self.device).unsqueeze(1)
		#### CRITIC LOSS ####
		with torch.no_grad():
			next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
			qf_next_target = self.critic_target(next_state_batch,next_state_action,subset_size=self.subset_size).min(0)[0] # take minimum across ensemble subset.  [0] selects values (as opposed to indecies).
			if not self.args.no_entropy_backup:
				# we are doing entropy backup
				qf_next_target -= self.alpha * next_state_log_pi
			next_q_value = reward_batch + mask_batch * self.gamma * qf_next_target

		qfs = self.critic(state_batch,action_batch) # q values for each ensemble member, n_members x batch_size x state_dim x 1.
		
		# ipdb.set_trace()
		qf_loss = self.QLoss(qfs,torch.stack(self.args.q_ensemble_members*[next_q_value]))
		if torch.any(torch.isnan(qf_loss)):
			print('qf loss is nan')

		self.critic_optim.zero_grad()
		qf_loss.backward()
		self.critic_optim.step()

		#### POLICY LOSS ####
		pi, log_pi, _ = self.policy.sample(state_batch)
		qf_pi = self.critic(state_batch, pi).mean(0) # take mean across ensemble members

		policy_loss = ((self.alpha * log_pi) - qf_pi).mean()

		if torch.any(torch.isnan(policy_loss)):
			print('policy loss is nan')

		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		#### ALPHA LOSS ####
		if self.automatic_entropy_tuning:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()

			self.alpha = self.log_alpha.exp()
			alpha_tlogs = self.alpha.clone() # For TensorboardX logs
		else:
			alpha_loss = torch.tensor(0.).to(self.device)
			alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

		self.updates += 1
		if self.updates % self.target_update_interval == 0:
			soft_update(self.critic_target, self.critic, self.tau)

		return qf_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), -torch.mean(log_pi).item()

	def save(self,path):
		torch.save({'policy_state_dict':self.policy.state_dict(),
					'policy_optimizer_state_dict':self.policy_optim.state_dict(),
					'critic_state_dict':self.critic.state_dict(),
					'critic_target_state_dict':self.critic_target.state_dict(),
					'critic_optimizer_state_dict':self.critic_optim.state_dict()
					},path)

	def load(self,path):
		ckpt = torch.load(path)

		self.policy.load_state_dict(ckpt['policy_state_dict'])
		self.policy_optim.load_state_dict(ckpt['policy_optimizer_state_dict'])
		self.critic.load_state_dict(ckpt['critic_state_dict'])
		self.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
		self.critic_optim.load_state_dict(ckpt['critic_optimizer_state_dict'])