#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function

import os
import base64
import imageio
import IPython
import reverb

import pandas as pd
import tensorflow as tf

from model.tetris import Tetris

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


# In[ ]:


# GAME SETTING
BOARD_SIZE_W, BOARD_SIZE_H = BOARD = (10, 20)

# SCREEN SETTING
BLOCK_SIZE_W, BLOCK_SIZE_H = BLOCK_SIZE = (24, 24)
PADDING_X, PADDING_Y = PADDING = (24, 24)
WINDOW_SIZE_W, WINDOW_SIZE_H = WINDOW_SIZE = (PADDING_X * 2 + BOARD_SIZE_W * BLOCK_SIZE_W, PADDING_Y * 2 + BOARD_SIZE_H * BLOCK_SIZE_H)
env = Tetris(WINDOW_SIZE, PADDING, BOARD, BLOCK_SIZE)
utils.validate_py_environment(env, episodes=20)


# In[ ]:


# Hyper Params
num_iterations = 100_000_000

collect_steps_per_iteration = 20
replay_buffer_max_length = 500_000

batch_size = 128
learning_rate = 5e-5
log_interval = 1_000

num_eval_episodes = 10
eval_interval = 1_000


# In[ ]:


# Load environment
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)


# In[ ]:


# Video functions
def embed_mp4(filename):
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


# In[ ]:


# Modelling
fc_layer_params = (32, 32)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.elu,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'),
        kernel_initializer=tf.keras.initializers.he_normal,# tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
        bias_initializer=tf.keras.initializers.he_normal, #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),
    )

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


# In[ ]:


# Init model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
)
agent.initialize()


# In[ ]:


# Metrics
def compute_avg_return(environment, policy, num_episodes=10):
  returns = []
  steps = []
  for _ in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0
    step = 0
    while not time_step.is_last():
      step += 1
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    steps.append(step)
    returns.append(episode_return.numpy()[0])
  return steps, (min(returns), sum(returns) / len(returns), max(returns))


# In[ ]:


table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
table = reverb.Table(
  table_name,
  max_size=replay_buffer_max_length,
  sampler=reverb.selectors.Uniform(),
  remover=reverb.selectors.Fifo(),
  rate_limiter=reverb.rate_limiters.MinSize(1),
  signature=replay_buffer_signature
)

reverb_server = reverb.Server([table])
replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
  agent.collect_data_spec,
  table_name=table_name,
  sequence_length=2,
  local_server=reverb_server
)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2
)


# In[ ]:


# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=8,
    sample_batch_size=batch_size,
    num_steps=2
).prefetch(8)


# In[ ]:


# Checkpointer and policy saver
def create_checkpointer(ckpt_dir, agent, replay_buffer, train_step_counter):
    return common.Checkpointer(
        ckpt_dir=ckpt_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step_counter,
    )


# In[ ]:


print(f'Initial Step: {agent.train_step_counter.numpy()}')

# Restore checkpoint Best
checkpointer_best = create_checkpointer('checkpoint/best', agent, replay_buffer, train_step_counter)
checkpointer_best.initialize_or_restore()
print(f'Best Step: {agent.train_step_counter.numpy()}')
create_policy_eval_video(agent.policy, f'checkpoint-best', fps=10)

# Restore checkpoint Latest
checkpointer_latest = create_checkpointer('checkpoint/latest', agent, replay_buffer, train_step_counter)
checkpointer_latest.initialize_or_restore()
print(f'Latest Step: {agent.train_step_counter.numpy()}')
create_policy_eval_video(agent.policy, f'checkpoint-latest', fps=10)


# In[ ]:


iterations_start = agent.train_step_counter.numpy()
iterations_end = iterations_start + num_iterations
print(f'iterations_start: {iterations_start}')
print(f'iterations_end: {iterations_end}')


# In[ ]:


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the environment.
collect_driver = py_driver.PyDriver(
  env,
  py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
  [rb_observer],
  max_steps=collect_steps_per_iteration,
  max_episodes=1,
)

# Metrics: returns
if os.path.exists('metrics-returns.csv'):
  df_returns = pd.read_csv('metrics-returns.csv')
else:
  df_returns = pd.DataFrame([], columns=['min', 'avg', 'max'])

# Metrics: steps
if os.path.exists('metrics-steps.csv'):
  df_steps = pd.read_csv('metrics-steps.csv')
else:
  df_steps = pd.DataFrame([], columns=['min', 'avg', 'max'])
  
# Load highest records
highest_avg = df_returns['avg'].max()
if pd.isna(highest_avg): highest_avg = 0
print(f'highest_avg: {highest_avg}')

# Start training
print(f'START TRAINING')
iterator = iter(dataset)
time_step = env.reset()
for _ in range(iterations_start, iterations_end):
  # Collect a few steps and save to the replay buffer.
  time_step, _ = collect_driver.run(time_step)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()
  if step % eval_interval == 0:
    # Evaluate
    total_steps, (ret_min, ret_avg, ret_max) = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    step_min, step_avg, step_max = min(total_steps), int(sum(total_steps)/len(total_steps)), max(total_steps)

    # Save metrics: return
    df_returns = pd.concat([df_returns, pd.DataFrame([[ret_min, ret_avg, ret_max]], columns=['min', 'avg', 'max'])])
    # df_returns = df_returns.append({'min': ret_min, 'avg': ret_avg, 'max': ret_max}, ignore_index = True)
    df_returns.to_csv('metrics-returns.csv', index=False)

    # Save metrics: steps
    df_steps = pd.concat([df_steps, pd.DataFrame([[step_min, step_avg, step_max]], columns=['min', 'avg', 'max'])])
    # df_steps = df_steps.append({'min': step_min, 'avg': step_avg, 'max': step_max}, ignore_index = True)
    df_steps.to_csv('metrics-steps.csv', index=False)

    # Print
    log = f'Iteration = {step} | Total steps: {min(total_steps)}-{int(sum(total_steps)/len(total_steps))}-{max(total_steps)} | Min: {ret_min:.02f} | Avg: {ret_avg:.02f} | Max: {ret_max:.02f}'
    print(log)

    # Save latest
    checkpointer_latest.save(train_step_counter)

    # Check new best record
    if ret_avg > highest_avg:
      highest_avg = ret_avg

      # Save Model
      model_name = f"iteration-{step:09d}"
      tf_policy_saver = policy_saver.PolicySaver(agent.policy)
      tf_policy_saver.save(f'wheights/{model_name}')
      print(f'Best policy saved')

      # Save Checkpoint
      checkpointer_best.save(train_step_counter)
      print(f'Best checkpoint saved')
      
      # Create video
      if not os.path.exists('videos'):
        os.mkdir('videos')
        print('Video folder created')
      create_policy_eval_video(agent.policy, f'videos/{model_name}', fps=10)
      print(f'Video saved')
    
    # Write a live text logs
    myfile = open('logs.txt', 'a')
    myfile.write(f"{log}")
    if highest_avg == ret_avg:
      myfile.write(f" [CHECK POINT]")
    myfile.write(f"\n")
    myfile.close()

