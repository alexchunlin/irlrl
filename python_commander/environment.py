
import numpy as np
import time 
import gym
from gym import spaces


from motorgo_echo import send_motor_command, receive_reply

class RobotEnv():
    def __init__(self,ep_len):

        # self.action_space = 
        self.ep_len = ep_len
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

    def reset(self):
        send_motor_command(0)
        _ = receive_reply()

        return

    def wait_till_reset(self):
        vel_is_zero = False
        while not vel_is_zero:
            send_motor_command(0)
            position, velocity = receive_reply()
            vel_is_zero = (velocity == 0)

        
        self.start_thyme = time.time()

        

        return np.array([np.sin(position),np.cos(position),velocity])


    def step(self,a):
        # t += 1
        print('a: ', a)
        a = a.item()
        send_motor_command(a)
        position, velocity = receive_reply()
        s = np.array([np.sin(position),np.cos(position),velocity])  

        # compute reward
        r = -angle_normalize(position - np.pi) ** 2 - 0.1 * velocity**2 - 0.001 * (a**2)

        # d = t == self._max_episode_steps
        elapsed_thyme = time.time() - self.start_thyme
        print('elapsed_thyme: ', elapsed_thyme)
        if elapsed_thyme >= self.ep_len:
            d = True
        else:
            d = False
    
        return s, r, d, None         

    def sample_action(self):
        elapsed_thyme = time.time() - self.start_thyme
        return np.array([np.sin(elapsed_thyme**2)])



def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi