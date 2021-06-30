import tensorflow as tf
from PIL import Image
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Conv2D,Flatten
import matplotlib.pyplot as plt

alpha=10.0
inception=InceptionV3(include_top=False,pooling="avg")
learning_rate=3e-4
margin=2.0

class CoordinateUtils(object):
  @staticmethod
  def get_image_coordinates(h, w, normalise):
    x_range = tf.range(w, dtype=tf.float32)
    y_range = tf.range(h, dtype=tf.float32)
    if normalise:
      x_range = (x_range / (w - 1)) * 2 - 1
      y_range = (y_range / (h - 1)) * 2 - 1
    image_x = tf.repeat(tf.expand_dims(x_range,0),h, 0)
    image_y = tf.transpose(tf.repeat(tf.expand_dims(y_range,0),w, 0))
    return image_x, image_y

class spatial_softmax(tf.keras.Model):
  def __init__(self, temperature=None, normalise=True):
    super(spatial_softmax,self).__init__()
    self.temperature = tf.ones(1) if temperature is None else tf.tensor([temperature])
    self.normalise = normalise

  def call(self,inputs):
    N,H,W,C = inputs.shape
    features = tf.reshape(tf.transpose(inputs, [0, 3, 1, 2]), [N * C, H * W])
    softmax = tf.nn.softmax(features)
    softmax = tf.reshape(softmax, [N, C, H, W])
    softmax = tf.expand_dims(softmax, -1)
    # Convert image coords to shape [H, W, 1, 2]
    image_x, image_y = CoordinateUtils.get_image_coordinates(H, W, normalise=self.normalise)
    image_coords = tf.concat([tf.expand_dims(image_x,-1), tf.expand_dims(image_y,-1)],-1)
    
    # Multiply (with broadcasting) and reduce over image dimensions to get the result
    # of shape [N, C, 2]
    spatial_soft_argmax = tf.reduce_sum(softmax * image_coords,[2, 3])
    return spatial_soft_argmax
 
def normalize(output_embedding):
  buffer = tf.math.pow(output_embedding, 2)
  normp = tf.math.reduce_sum(buffer, 1,keepdims=True) + 1e-10
  normalization_constant = tf.math.sqrt(normp)
  output = tf.math.divide(output_embedding, normalization_constant)
  return output

def get_embeddings(input_frame):
  input_frame=preprocess_input(input_frame)
  output_embedding=inception.predict(input_frame)
  return output_embedding

class TCNModel(tf.keras.Model):
  def __init__(self,temperature=None, normalise=True):
    super(TCNModel,self).__init__()

    self.layer_1 = Conv2D(100,kernel_size=3,strides=1,activation='relu')
    self.layer_2 = Conv2D(20,kernel_size=3,strides=1,activation='relu')
    self.layer_3 = spatial_softmax(temperature=None, normalise=True)
    self.layer_4 = Flatten()
    self.layer_5 = Dense(32, activation='relu')

  def call(self,inputs):
    x =  self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(inputs)))))
    x = normalize(x) * alpha
    return x

def ls(path):
    # returns list of files in directory without hidden ones.
    return [p for p in os.listdir(path) if p[0] != '.']

def _resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize((299,299))
    scaled = np.array(image, dtype=np.float32) / 255
    return scaled


def read_video(filepath, frame_size):
    imageio_video = imageio.read(filepath)
    snap_length = len(imageio_video)
    frames = np.zeros((snap_length,*frame_size))
    for i, frame in enumerate(imageio_video):
        frame=_resize_frame(frame, frame_size)
        frames[i, :, :, :] = frame
    return frames

class SingleViewTripletBuilder(object):
    def __init__(self, video_directory, image_size):
        self.frame_size = image_size
        self._read_video_dir(video_directory)

        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of the frame.
        self.positive_frame_margin = 10
        self.negative_frame_margin = 30
        self.video_index = 0

    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory

        self._anchor_directory=os.path.join(self._video_directory, 'Anchor')
        self._positive_directory=os.path.join(self._video_directory, 'Positive')

        anchor_filenames = ls(self._anchor_directory)
        positive_filenames = ls(self._positive_directory)
  
        self.anchor_video_paths = [os.path.join(self._anchor_directory, f) for f in anchor_filenames]
        self.positive_video_paths = [os.path.join(self._positive_directory, f) for f in positive_filenames]

        self.video_count = len(self.anchor_video_paths)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.anchor_video_paths])
        self.frame_lengths = frame_lengths

    def get_video(self, index):
        print(self.anchor_video_paths[index],self.positive_video_paths[index])
        return (read_video(self.anchor_video_paths[index], self.frame_size),read_video(self.positive_video_paths[index], self.frame_size))

    def sample_triplet(self, anchor_video,positive_video):
        anchor_index = self.sample_anchor_frame_index()
        positive_index = anchor_index
        negative_index = self.sample_negative_frame_index(anchor_index)
        anchor_frame = anchor_video[anchor_index]
        positive_frame = positive_video[positive_index]
        negative_frame = anchor_video[negative_index]
        return (anchor_frame,positive_frame,negative_frame)

    def build_set(self,batch_size):
        triplets = []
        triplets = np.zeros((batch_size,3,299,299,3))
        for i in range(0, batch_size):
            anchor_video,positive_video = self.get_video(self.video_index)
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(anchor_video,positive_video)

            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame
            fig = plt.figure(figsize=(12,12))
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(anchor_frame)
            ax2 = fig.add_subplot(2,2,2)
            ax2.imshow(positive_frame)
            ax3 = fig.add_subplot(2,2,3)
            ax3.imshow(negative_frame)
            plt.show()
            self.video_index = (self.video_index + 1) % self.video_count
        # Second argument is labels. Not used.
        return triplets

    def sample_anchor_frame_index(self):
        arange = np.arange(0, self.frame_lengths[self.video_index])
        return np.random.choice(arange)

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths[self.video_index]
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))

def distance(x1, x2):
  diff = ((abs(x1 - x2)**2).sum(dim=1))
  return diff

def validate(tcn):
    # Run model on validation data and log results
    tcn=TCNModel()
    IMAGE_SIZE = (299, 299,3)
    triplet_builder =  SingleViewTripletBuilder("Validation", IMAGE_SIZE)
    batch_size=64
    validation_dataset = triplet_builder.build_set(batch_size)

    correct_with_margin = 0
    correct_without_margin = 0

    anchor_frames = validation_dataset[:, 0, :, :, :]
    positive_frames = validation_dataset[:, 1, :, :, :]
    negative_frames = validation_dataset[:, 2, :, :, :]

    anchor_embeddings = get_embeddings(anchor_frames)
    positive_embeddings = get_embeddings(positive_frames)
    negative_embeddings = get_embeddings(negative_frames)

    anchor_output = tcn(anchor_embeddings)
    positive_output = tcn(positive_embeddings)
    negative_output = tcn(negative_embeddings)

    d_positive = distance(anchor_output, positive_output)
    d_negative = distance(anchor_output, negative_output)
    
    for i in range(64):
        print('D_positive',d_positive[i],'D_negative',d_negative[i])
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(anchor_frames[i])
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(positive_frames[i])
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(negative_frames[i])
        plt.show()

    correct_with_margin +=1*((d_positive + margin) < d_negative)
    correct_without_margin += 1*(d_positive < d_negative)
    
    print("Validation score correct with margin {with_margin}/{total} and without margin {without_margin}{total}".format(with_margin=tf.reduce_sum(correct_with_margin),without_margin=tf.reduce_sum(correct_without_margin),total=len(validation_dataset)))


def train_step(anchor_embeddings,positive_embeddings,negative_embeddings,tcn):
  optimizer_policy = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  with tf.GradientTape() as tape:
    anchor_output = tcn(anchor_embeddings)
    positive_output = tcn(positive_embeddings)
    negative_output = tcn(negative_embeddings)
    d_positive = distance(anchor_output, positive_output)
    d_negative = distance(anchor_output, negative_output)
    loss=margin + d_positive - d_negative
  grads=tape.gradient(loss,tcn.trainable_variables)
  optimizer_Q_function.apply_gradients(zip(grads, tcn.trainable_variables))
  return loss

def train_tcn():
  tcn=TCNModel()
  epochs=1000
  IMAGE_SIZE = (299, 299,3)
  triplet_builder =  SingleViewTripletBuilder("Training", IMAGE_SIZE)
  ITERATE_OVER_TRIPLETS = 5
  batch_size=128

  for epoch in range(epochs):
    training_dataset = triplet_builder.build_set(batch_size)
    anchor_frames = training_dataset[:, 0, :, :, :]
    positive_frames = training_dataset[:, 1, :, :, :]
    negative_frames = training_dataset[:, 2, :, :, :]

    anchor_embeddings = get_embeddings(anchor_frames)
    positive_embeddings = get_embeddings(positive_frames)
    negative_embeddings = get_embeddings(negative_frames)
    loss = 0
    if epoch%100==0 and epoch!=0 and learning_rate>3e-5:
      learning_rate=learning_rate*0.1
    for _ in range(0, ITERATE_OVER_TRIPLETS):
      loss+=train_step(anchor_embeddings,positive_embeddings,negative_embeddings,tcn)
    print("Epoch:{} Loss:{}".format(epoch,np.mean(loss)))
    tcn.save('tcn_{}.h5'.format(epoch))
    if epoch % 10 == 0 and epoch!=0:
      validate(tcn)
      
train_tcn()
