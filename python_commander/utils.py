import numpy as np
import torch
import gym
import os
# import d4rl
import ipdb



def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

def numpify(x):
	return x.detach().cpu().numpy()

def get_mean_std(x):

	x_reshaped = x.reshape(-1,x.shape[-1])
	mean = np.mean(x_reshaped,axis=0)
	std = np.std(x_reshaped,axis=0)

	return mean,std

def get_gaussian_ll(mean,std,x):
	'''
	Returns log likelihood of x according to (independent) Gaussians with mean and std	
	'''
	return -torch.log(std) - .5*np.log(2*np.pi) - .5* ((x-mean)**2)/(std**2)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', default='Walker2d-v3')
	parser.add_argument('--policy_type', default="tanh_gaussian")
	parser.add_argument('--ep_len', type=int, default=100)
	parser.add_argument('--no_termination',action='store_true')
	parser.add_argument('--sparse',action='store_true')
	parser.add_argument('--vel_thresh',type=float,default=1.0)
	parser.add_argument('--distractors',action='store_true')
	parser.add_argument('--n_groups',type=int,default=10)
	parser.add_argument('--n_per_group',type=int,default=10)
	parser.add_argument('--done_zero_reward',action='store_true')
	args = parser.parse_args()

	args.done_zero_reward = True

	env = make_env(args)
	print('env: ', env)

	for j in range(3):
		print('===============================')
		state = env.reset()
		done = False
		i = 0
		last_r = None
		while not done:
			state,r,done,_ = env.step(env.action_space.sample())
			print('r: ', r)
			i += 1
			print('i: ', i)
			print('j: ', j)
			if last_r == 0 and r != 0:
				ipdb.set_trace()

			last_r = r


def get_auc_mean_std_err(x_list,y_list,x_shift=0,x_lim=[100,9990]):
	'''
	x_list: N x T numpy array
	y_list: N x T numpy array
	
	x_lim: 2-length list containing bounds for integration

	
	'''
	auc_list = []
	for i in range(len(y_list)):
		# trim to fit within x_lims
		# ipdb.set_trace()
		x_list_i = np.array(x_list[i]) + x_shift
		# make sure we're within bounds
		try:
			assert x_lim[0] >= x_list_i[0]
			assert x_lim[1] <= x_list_i[-1]
			# print('min x_list_i: ',np.min(x_list_i))
			# print('max x_list_i: ',np.max(x_list_i))
			y_list_i = np.array(y_list[i])
			inds_keep = (x_list_i >= x_lim[0]) & (x_list_i <= x_lim[1])
			x_list_i = x_list_i[inds_keep]
			y_list_i = y_list_i[inds_keep]
			auc_i = np.trapz(y_list_i,x=x_list_i) / (x_list_i[-1] - x_list_i[0])
			auc_list.append(auc_i)
		except:
			print('WARNING! Sequence ',i, ' might not be long enough')
			# pass
			# ipdb.set_trace()
		
	# ipdb.set_trace()
	auc_mean = np.mean(auc_list)
	auc_std_err = 1.96*np.std(auc_list)/len(auc_list)
	
	# print('auc: $', "{:e}".format(auc_mean), ' \pm ',"{:e}".format(auc_std_err),' $')
	print('$', " {:.1f}".format(auc_mean), ' \pm '," {:.1f}".format(auc_std_err),' $')

