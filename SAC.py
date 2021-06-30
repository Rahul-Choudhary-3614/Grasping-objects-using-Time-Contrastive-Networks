# Commented out IPython magic to ensure Python compatibility.
#Importing libraries
import tensorflow as tf
from PIL import Image
import imageio
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
from tensorflow.keras.utils import normalize as normal_values
gymlogger.set_level(40) #error only
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import math
import glob
import io
import tensorflow_probability as tfp
import base64
from IPython.display import HTML
from IPython.display import clear_output
from IPython import display as ipythondisplay
from collections import deque
import os
import numpy as np
import random
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.keras.models import load_model,Model,Sequential
from tensorflow.keras.layers import Dense,Dropout,Input,Concatenate
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
import robot_2
import pybullet
from pyvirtualdisplay import Display

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

env_=robot_2.KukaDiverseObjectEnv()
upper_bound = env_.action_space.high[0]
lower_bound = env_.action_space.low[0]

inception=InceptionV3(include_top=False, pooling="avg")

class GaussianPolicy():
  def __init__(self,action_dim,state_dim ,reparameterize,path=None):
    self.reparameterize = reparameterize
    self.action_dim=action_dim
    self.state_dim=state_dim
    if path is None:
        self.model = self.create_model()
    else:
        self.model = load_model(path)

  def create_model(self):
    model=Sequential()
    model.add(Input(shape=self.state_dim))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(self.action_dim * 2, activation=None))
    return model

  def get_action(self, inputs):
    mean_and_log_std = self.model(inputs)
    mean, log_std = tf.split(mean_and_log_std, num_or_size_splits=2, axis=1)
    log_std = tf.clip_by_value(log_std, -20., 2.)
    
    distribution = tfp.distributions.MultivariateNormalDiag(loc=mean,scale_diag=tf.exp(log_std))
    
    raw_actions = distribution.sample()
    if not self.reparameterize:
        raw_actions = tf.stop_gradient(raw_actions)
    log_probs = distribution.log_prob(raw_actions)
    log_probs -= self._squash_correction(raw_actions)

    self.actions = tf.tanh(raw_actions)
    
    return self.actions, log_probs
    
  def _squash_correction(self, raw_actions, stable=True, eps=1e-8):
    if not stable:
      return tf.reduce_sum(tf.math.log(1. - tf.square(tf.tanh(raw_actions)) + eps), axis=1)
    else:
      return tf.reduce_sum(tf.math.log(4.) + 2. * (raw_actions - tf.nn.softplus(2. * raw_actions)), axis=1)

  def eval(self, state):
        
    mean_and_log_std = self.model(state)
    mean, log_std = tf.split(mean_and_log_std, num_or_size_splits=2, axis=1)
    return mean.numpy()

def _create_Q_function(action_shape,state_shape):
  state_input = Input(shape=state_shape)
  action_input = Input(shape=(action_shape,))

  layer_1=(Dense(1024, activation="relu"))(state_input)  
  layer_2=(Dense(512, activation="relu"))(layer_1)
  layer_3=(Dense(512, activation="relu"))(layer_2)  

  action_layer_1 = Dense(512, activation="relu")(action_input)
  concat = Concatenate()([layer_3,action_layer_1])
  concat_layer_1=Dense(512,activation="relu")(concat)
  
  output = Dense(1, activation=None)(concat_layer_1)
  model = Model(inputs=[state_input,action_input],outputs=[output])

  return model


def policy_loss(states):
  if  reparameterize:
    actions, log_pis = policy.get_action(states)
    #print(actions.shape)
    if Q_2_function is None:
      q_n = Q_1_function((states, actions))
      #print(q_n.shape)
    else:
      q_n = tf.minimum(Q_1_function((states, actions)),Q_2_function((states, actions)))                
      #print(q_n.shape)
    policy_loss = tf.reduce_mean(alpha * log_pis - q_n)
    #print(policy_loss.shape)

  else:
    actions, log_pis = policy.get_action(states)
    #print(actions.shape,log_pis.shape)
    if Q_2_function is None:
      q_n = Q_1_function((states, actions))
      #print(q_n.shape)
    else:
      q_n = tf.minimum(Q_1_function((states, actions)),Q_2_function((states, actions)))                
      print(q_n.shape)
    policy_loss = tf.reduce_mean(log_pis * tf.stop_gradient(alpha * log_pis - q_n))
    #print(policy_loss.shape)
  #print('Policy Loss',policy_loss)
  return policy_loss
  
def q_1_function_loss(states,next_states,actions,dones,rewards):
  q_n = Q_1_function((states, actions))

  next_states_actions, next_states_actions_log_pis = policy.get_action(next_states)
  #print(next_states_actions.shape)
  
  next_q_n_1 = target_Q_1_function((next_states,next_states_actions))
  next_q_n_2 = target_Q_2_function((next_states,next_states_actions))
  target_q_n= rewards + (1 - dones) * gamma * (tf.minimum(next_q_n_1,next_q_n_2)-next_states_actions_log_pis*alpha)
  q_function_loss = tf.reduce_mean(tf.losses.mean_squared_error(target_q_n,q_n))
  #print("Q_1_function_loss",q_function_loss)
  return q_function_loss 

def q_2_function_loss(states,next_states,actions,dones,rewards):
  q_n = Q_2_function((states, actions))

  next_states_actions, next_states_actions_log_pis = policy.get_action(next_states)
  #print(next_states_actions.shape)
  
  next_q_n_1 = target_Q_1_function((next_states,next_states_actions))
  next_q_n_2 = target_Q_2_function((next_states,next_states_actions))
  target_q_n= rewards + (1 - dones) * gamma * (tf.minimum(next_q_n_1,next_q_n_2)-next_states_actions_log_pis*alpha)
  q_function_loss = tf.reduce_mean(tf.losses.mean_squared_error(target_q_n,q_n))
  #print("Q_2_function_loss",q_function_loss)
  return q_function_loss

class Buffer:
  def __init__(self, buffer_capacity=100000, batch_size=64):
    # Number of "experiences" to store at max
    self.buffer_capacity = buffer_capacity
    # Num of tuples to train on.
    self.batch_size = batch_size

    # Its tells us num of times record() was called.
    self.buffer_counter = 0

    # Instead of list of tuples as the exp.replay concept go
    # We use different np.arrays for each tuple element
    self.state_buffer = np.zeros((self.buffer_capacity, *state_shape))
    self.action_buffer = np.zeros((self.buffer_capacity, action_shape))
    self.reward_buffer = np.zeros((self.buffer_capacity, 1))
    self.next_state_buffer = np.zeros((self.buffer_capacity, *state_shape))
    self.done_buffer = np.zeros((self.buffer_capacity, 1))

  # Takes (s,a,r,s') obervation tuple as input
  def record(self, obs_tuple):
    
    # Set index to zero if buffer_capacity is exceeded,
    # replacing old records
    index = self.buffer_counter % self.buffer_capacity

    self.state_buffer[index] = obs_tuple[0]
    self.action_buffer[index] = obs_tuple[1]
    self.reward_buffer[index] = obs_tuple[2]
    self.next_state_buffer[index] = obs_tuple[3]
    self.done_buffer[index] = obs_tuple[4]

    self.buffer_counter += 1

    # We compute the loss and update parameters
  def learn(self):
    # Get sampling range
    record_range = min(self.buffer_counter, self.buffer_capacity)
    # Randomly sample indices
    batch_indices = np.random.choice(record_range, self.batch_size)

    # Convert to tensors
    state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
    action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
    reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
    next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
    done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])

    # Training and updating Actor & Critic networks.
    
    # Q_1 update
    with tf.GradientTape() as tape:
      critic_loss = q_1_function_loss(tf.cast(state_batch,tf.float32),tf.cast(next_state_batch,tf.float32),tf.cast(action_batch,tf.float32),tf.cast(done_batch,tf.float32),tf.cast(reward_batch,tf.float32))
      critic_grad = tape.gradient(critic_loss, Q_1_function.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, Q_1_function.trainable_variables)
    )
    # Q_2 update
    with tf.GradientTape() as tape:
      critic_loss = q_2_function_loss(tf.cast(state_batch,tf.float32),tf.cast(next_state_batch,tf.float32),tf.cast(action_batch,tf.float32),tf.cast(done_batch,tf.float32),tf.cast(reward_batch,tf.float32))
      critic_grad = tape.gradient(critic_loss, Q_2_function.trainable_variables)
      critic_optimizer.apply_gradients(zip(critic_grad, Q_2_function.trainable_variables))

    # Policy update
    with tf.GradientTape() as tape:
      actor_loss = policy_loss(state_batch)
      actor_grad = tape.gradient(actor_loss, policy.model.trainable_variables)
      actor_optimizer.apply_gradients(zip(actor_grad, policy.model.trainable_variables))

def update_target(tau):
  new_weights = []
  target_variables = target_Q_1_function.weights
  for i, variable in enumerate(Q_1_function.weights):
    new_weights.append(variable * tau + target_variables[i] * (1 - tau))

  target_Q_1_function.set_weights(new_weights)

  new_weights = []
  target_variables = target_Q_2_function.weights
  for i, variable in enumerate(Q_2_function.weights):
    new_weights.append(variable * tau + target_variables[i] * (1 - tau))

  target_Q_2_function.set_weights(new_weights)

def get_state(state):
  state=state.reshape((299,299,3))
  state=np.expand_dims(state,axis=0)
  output_embedding=inception.predict(state)
  state_=tcn(output_embedding)
  return state_
  
def _resize_frame(frame):
  image = Image.fromarray(frame)
  image = image.resize((299,299))
  scaled = np.array(image, dtype=np.float32) / 255
  return scaled

def get_reward(Expert_state,Policy_state):
  alpha=0.1
  beta=0.1
  diff=(Expert_state-Policy_state)**2
  diff=np.sum(diff,axis=-1)
  reward=-diff*alpha-beta*((diff+1e-12)**(1/2))
  #reward=np.exp(-(Expert_state-Policy_state))
  #reward=np.sum(reward,axis=-1)
  return reward[0]

def get_frames(filepath):
  frame_size = (299,299,3)
  imageio_video = imageio.read(filepath)
  snap_length = len(imageio_video)
  frames = np.zeros((snap_length,*frame_size))
  for i, frame in enumerate(imageio_video):
    frame=_resize_frame(frame)
    frames[i, :, :, :] = frame
  return frames,snap_length

reparameterize=True
gamma=[0.99] # decay rate of past observations
tau=0.01
action_shape=env_.action_space.shape[0]
state_shape=(2048,)
epsilon=1.0
batch_size=64
alpha=0.2 #Entropy Loss ratio
critic_lr = 3e-4
actor_lr = 3e-4
total_episodes=1200
videopath="Picking up the ball.mp4"
frames,T_max=get_frames(videopath)
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
tcn=load_model("tcn_20.h5")

policy=GaussianPolicy(action_shape,state_shape,reparameterize)
Q_1_function=_create_Q_function(action_shape,state_shape)
Q_2_function=_create_Q_function(action_shape,state_shape)    
target_Q_1_function=_create_Q_function(action_shape,state_shape)
target_Q_2_function=_create_Q_function(action_shape,state_shape)
buffer = Buffer(50000, batch_size)

def evaluate():
  print("Evaluating")
  env=wrap_env(robot_2.KukaDiverseObjectEnv())
  state=env.reset()
  done=False
  T_max=500
  state_1=get_state(state)
  episode_reward_original=0  
  for t in range(1,T_max):
    action=policy.eval(state_1)
    next_state, reward, done, info=env.step(action[0])
    next_state_1=get_state(next_state)
    if done:
      break
    state_1=next_state_1
    episode_reward_original+=reward
  env.close()
  print('Reward:{} Sucess:{}'.format(episode_reward_original,info['grasp_success']))
  #env.render("ipython")
  #show_video()

list_=list(range(11,46))
images_list=[1,2,9,10]
images_list.extend(list_)

j=0
for i,frame in enumerate(frames):
  if i+1 in images_list:
    frames_[j]=frame
    j+=1

# Takes about 20 min to train
ep_reward_list=[]
avg_reward_list = []
T_max=39
total_episodes=1200
for ep in range(total_episodes):
  try:
    state=env_.reset()  
  except:
    pybullet.connect(pybullet.DIRECT)
    state=env_.reset()
  done=False
  episodic_reward = 0
  state_1=get_state(state)
  expert_state_1=get_state(frames_[0])
  episode_reward=0
  for t in range(1,T_max):
    action,_=policy.get_action(state_1)
    # Recieve state and reward from environment.
    next_state, reward, done, info = env_.step(action[0])
    reward=5*get_reward(expert_state_1,state_1)
    if  info['grasp_success']:
      reward+=5
    next_state_1=get_state(next_state)
    buffer.record((state_1,action,reward,next_state_1,done))
    episodic_reward += reward

    buffer.learn()
    update_target(tau)

    # End this episode when `done` is True
    if done or t==T_max:
      break
    #print(t)
    state_1=next_state_1
    expert_state_1=get_state(frames_[t])
    #print(action,reward)

    #fig = plt.figure()
    #ax1 = fig.add_subplot(2,2,1)
    #ax1.imshow(frames_[t])
    #ax2 = fig.add_subplot(2,2,2)
    #ax2.imshow(next_state/255.)
    #plt.show()
  if ep%20==0 and ep!=0:
    evaluate()
  ep_reward_list.append(episodic_reward)
  # Mean of last 40 episodes
  avg_reward = np.mean(ep_reward_list[-40:])
  print("Episode * {} * Avg Reward * {} * Sucess is ==> {}".format(ep, avg_reward,info['grasp_success']))
  avg_reward_list.append(avg_reward)
  if ep%50==0 and ep!=0:
    policy.model.save('picker_sac_policy_{}.h5'.format(ep))
    Q_1_function.save('picker_sac_Q_1_{}.h5'.format(ep))
    Q_2_function.save('picker_sac_Q_2_{}.h5'.format(ep))  
    target_Q_1_function.save('picker_sac_target_Q_1_{}.h5'.format(ep))
    target_Q_2_function.save('picker_sac_target_Q_2_{}.h5'.format(ep))
# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()