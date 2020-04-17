from PIL import Image

import gym
import pddlgym
from pddlgym.parser import parse_plan_step

from collections import defaultdict
from pddlgym.utils import VideoWrapper

import itertools
import numpy as np
import os
import gym
import imageio
import time

def run_render(env):
    # Output video (gif) to /tmp/
    outdir = "output"
    if not os.path.exists(outdir):
            os.makedirs(outdir)

    # Rendering config:
    video_path = os.path.join(outdir, 'random_{}_demo.gif'.format(env.spec.id))
    env = VideoWrapper(env, video_path, fps=5)

    obs, _ = env.reset()

    n = 10
    start_time = time.time()
    for t in range(n):
        #print("Obs:", obs)

        action = env.action_space.sample()
        print("Action:", action)

        obs, reward, done, _ = env.step(action)
        env.render()
        #print("Rew:", reward)

        if done:
            break

    end_time = time.time()
    print("Average time per action: ", str((end_time - start_time) / n), " seconds")
    env.close()

def run_norender_timing(env):
    obs, _ = env.reset()

    n = 10
    start_time = time.time()
    for t in range(n):
        #print("Obs:", obs)

        action = env.action_space.sample()
        #print("Action:", action)

        obs, reward, done, _ = env.step(action)
        #print("Rew:", reward)

        if done:
            break

    #print("Final obs:", obs)
    end_time = time.time()
    print("Average time per action: ", str((end_time - start_time) / n), " seconds")

    env.close()

if __name__ == '__main__':
    # Config gym
    env = gym.make("PDDLEnvEli-rubiks-attempt-v0")
    env.fix_problem_index(0) # Problem index is just a way to specify a specific pddl problem file

    #run_norender_timing(env)
    run_render(env)