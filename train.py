#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import numpy as np
import PIL.Image
from model.tetris import Tetris
from model.board import Board


import base64
import imageio
import IPython
import pyvirtualdisplay
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment


# In[2]:


# GAME SETTING
BOARD_SIZE_W, BOARD_SIZE_H = BOARD = (10, 20)

# SCREEN SETTING
BLOCK_SIZE_W, BLOCK_SIZE_H = BLOCK_SIZE = (24, 24)
PADDING_X, PADDING_Y = PADDING = (24, 24)
WINDOW_SIZE_W, WINDOW_SIZE_H = WINDOW_SIZE = (PADDING_X * 2 + BOARD_SIZE_W * BLOCK_SIZE_W, PADDING_Y * 2 + BOARD_SIZE_H * BLOCK_SIZE_H)
env = Tetris(WINDOW_SIZE, PADDING, BOARD, BLOCK_SIZE)
PIL.Image.fromarray(env.render())


# In[3]:


ret = env.board.take_action(0)
print(f'ret: {ret}')
PIL.Image.fromarray(env.render())


# In[4]:


env.board.state


# In[5]:


# raise Exception('STOP!')


# In[6]:


num_iterations = 10_000 # @param {type:"integer"}

collect_steps_per_iteration = 20 # @param {type:"integer"}
replay_buffer_max_length = 1_000_000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 2e-4  # @param {type:"number"}
log_interval = 1000  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


# In[7]:


from tf_agents.environments import utils

utils.validate_py_environment(env, episodes=20)


# In[8]:


train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)


# In[9]:


def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

def create_policy_eval_video(policy, filename, num_episodes=5, fps=15):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(env.render())
  return embed_mp4(filename)


# In[10]:


fc_layer_params = (256, 256, 128, 128, 64)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.elu,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'),
        kernel_initializer=tf.keras.initializers.he_normal,# tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
        bias_initializer=tf.keras.initializers.he_normal, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
    )

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    # activation=tf.keras.activations.,
    activation=None,
    # kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
    # bias_initializer=tf.keras.initializers.Constant(-0.2)
    # kernel_initializer=tf.keras.initializers.he_normal, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
    # bias_initializer=tf.keras.initializers.he_normal, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
)
q_net = sequential.Sequential(dense_layers + [q_values_layer])


# In[11]:


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

agent.initialize()


# In[12]:


eval_policy = agent.policy
collect_policy = agent.collect_policy


# In[13]:


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


# In[14]:


def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics


# In[15]:


compute_avg_return(eval_env, random_policy, num_eval_episodes)


# In[16]:


table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)


# In[17]:


# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=8,
    sample_batch_size=batch_size,
    num_steps=2
).prefetch(8)


# In[18]:


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
  env,
  py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
  [rb_observer],
  max_steps=collect_steps_per_iteration,
  max_episodes=1
)

num_iterations = 200_000
iterator = iter(dataset)
for _ in range(num_iterations):

  # Collect a few steps and save to the replay buffer.
  time_step, _ = collect_driver.run(time_step)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)


# In[19]:


PIL.Image.fromarray(env.render())


# In[20]:


# iterations = range(0, num_iterations + 1, eval_interval)
# plt.plot(iterations, returns)
# plt.ylabel('Average Return')
# plt.xlabel('Iterations')
# plt.ylim(top=250)


# In[25]:


create_policy_eval_video(agent.policy, "trained-agent", fps=10)


# In[26]:


create_policy_eval_video(random_policy, "random-agent")


# In[28]:


import os
from tf_agents.policies import policy_saver

tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save('wheights/mark-001')

