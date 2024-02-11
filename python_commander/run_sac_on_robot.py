from comet_ml import Experiment

import argparse
from datetime import datetime
# import gym
# import robel
import numpy as np
import itertools
import torch
from sac_agent import SAC
from data_buffer import ReplayMemory
# from wrappers import EpLengthWrapper, DistractorWrapper
import ipdb
# from utils import make_gif,make_env
from environment import RobotEnv
# from utils import evaluate_sac_policy

parser = argparse.ArgumentParser()
parser.add_argument('--policy_type', default="tanh_gaussian")
parser.add_argument('--mf_std_scale',type=float,default=1.0)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--automatic_entropy_tuning', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=1000001)
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--updates_per_step', type=int, default=1)
parser.add_argument('--start_steps', type=int, default=10000)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--eval_freq',type=int,default=10)
parser.add_argument('--deterministic_eval',action='store_true')
parser.add_argument('--n_eval_episodes',type=int,default=10)
parser.add_argument('--replay_size', type=int, default=1000000)
parser.add_argument('--q_ensemble_members', type=int, default=2)
parser.add_argument('--q_subset_size',type=int,default=2)
parser.add_argument('--q_layer_norm',action='store_true')
parser.add_argument('--no_entropy_backup',action='store_true')
parser.add_argument('--ep_len', type=int, default=None)
parser.add_argument('--save_gif',action='store_true')
parser.add_argument('--save_policy',action='store_true')
parser.add_argument('--no_termination',action='store_true')
parser.add_argument('--target_entropy',type=float,default=1.0,help='Sets the scale factor on the target entropy for automatic ent tuning')
# parser.add_argument('--huber',action='store_true')
# parser.add_argument('--sparse',action='store_true')
parser.add_argument('--vel_thresh',type=float,default=1.0)
parser.add_argument('--cpu',action='store_true')
# parser.add_argument('--save_data',action='store_true')
# parser.add_argument('--load_data',action='store_true')
# parser.add_argument('--data_file_name',type=str,default=None)
# parser.add_argument('--distractors',action='store_true')
# parser.add_argument('--n_groups',type=int,default=10)
# parser.add_argument('--n_per_group',type=int,default=10)
# parser.add_argument('--done_zero_reward',action='store_true')
args = parser.parse_args()

dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

args.alpha_mf = args.alpha  # for compatibility with URL code

env = RobotEnv(args.ep_len) #make_env(args)

agent = SAC(3, env.action_space, args)

experiment = Experiment('9mxH2vYX20hn9laEr0KtHLjAa',project_name="sac_on_robot")
experiment.log_parameters(vars(args))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# see if we should load data
# if args.data_file_name is None:
#     data_path = 'data/sac_buffer_'+args.env_name+'_ep_len_'+str(args.ep_len)
# else:
#     data_path = 'data/'+args.data_file_name
# if args.load_data:
#     memory.load_buffer(data_path)

# Training Loop
total_numsteps = 0
updates = 0

env.reset()
state = env.wait_till_reset()


for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    

    state = env.wait_till_reset()    
    
    while not done:
        # if we still have not collected enough random episodes of data, sample an action at random.
        if args.start_steps > total_numsteps: # total_numsteps is the total number of steps we've run so far in this episode.  
            action = env.sample_action()  # Sample random action
        # otherwise, sample action from policy
        else:
            action = agent.select_action(state)
            print('sampling from policy!')

        
            
        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        # print('episode_steps: ', episode_steps)
        total_numsteps += 1
        episode_reward += reward

        # make sure this is actually what we should be doing
        # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        mask = float(not done)
        print('mask: ', mask)
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    
    #########################
    
    env.reset()

    critic_losses = []
    policy_losses = []
    entropy_losses = []
    alphas = []
    entropies = []
    if len(memory) > args.batch_size:
        # Number of updates per step in environment
        for i in range(args.updates_per_step):
            # Update parameters of all the networks
            critic_loss, policy_loss, ent_loss, alpha, ent = agent.update_parameters(memory, args.batch_size)

            critic_losses.append(critic_loss)
            policy_losses.append(policy_loss)
            entropy_losses.append(ent_loss)
            alphas.append(alpha)
            entropies.append(ent)
            
            updates += 1
    
    
    
    #############################


    experiment.log_metric('loss/critic_1', np.mean(critic_losses), updates)
    experiment.log_metric('loss/policy', np.mean(policy_losses), updates)
    experiment.log_metric('loss/entropy_loss', np.mean(entropy_losses), updates)
    experiment.log_metric('entropy_temprature/alpha', np.mean(alphas), updates)
    experiment.log_metric('entropy',np.mean(entropies),updates)
                

    if total_numsteps > args.num_steps:  # this is when we're TOTALLY done with learning
        break

    experiment.log_metric('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # evaluation
    # if i_episode % args.eval_freq:
    #     # avg_reward = evaluate_sac_policy(env,agent,args.n_eval_episodes,deterministic=args.deterministic_eval,save_gif=args.save_gif,name='sac_'+args.env_name+'_'+str(args.ep_len)+'_'+dt)
    #     experiment.log_metric('mean_rew', avg_reward, i_episode)

    #     if args.save_data:
    #         memory.save_buffer(data_path)

    #     if args.save_policy:
    #         path = 'policies/'+'sac_policy_'+args.env_name + '_ep_len_' + str(args.ep_len)
    #         torch.save(agent.policy.state_dict(), path)