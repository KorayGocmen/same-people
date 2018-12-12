import os, itertools, random, imageio, sklearn
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

faces_folder = "faces"
pickle_file = "faces.pickle"
trained_models_folder = "./tf_trained/"

pixel_depth = 255.0
image_size = 128
num_labels = 2
input_size = image_size * image_size * 2

n_bytes = 2**31
max_bytes = 2**31 - 1
bytes_in = bytearray(0)

pickle_file_size = os.path.getsize(pickle_file)
with open(pickle_file, 'rb') as f_in:
  for _ in range(0, pickle_file_size, max_bytes):
    bytes_in += f_in.read(max_bytes)
save = pickle.loads(bytes_in)

train_dataset = save['train_dataset']
train_labels = save['train_labels']
valid_dataset = save['valid_dataset']
valid_labels = save['valid_labels']
test_dataset = save['test_dataset']
test_labels = save['test_labels']
del save  # hint to help gc free up memory

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = sklearn.utils.shuffle(train_dataset, train_labels)
valid_dataset, valid_labels = sklearn.utils.shuffle(valid_dataset, valid_labels)
test_dataset, test_labels = sklearn.utils.shuffle(test_dataset, test_labels)
print("done shuffle")

def accuracy(predictions, labels):
  compare_elements = np.argmax(predictions, 1) == np.argmax(labels, 1)
  return (100.0 * np.sum(compare_elements) / predictions.shape[0])

def makedir(path):
  if not os.path.exists(path):
    os.makedirs(path)
    
def accuracy_averaged(steps, accuracy_over_time, every_index=10):
  new_accuracy = np.mean(np.array(accuracy_over_time).reshape(-1, every_index), axis=1)
  new_steps = steps[0::every_index]
  return new_steps, new_accuracy


model_name = "3HiddenDropoutAndL2Optimization"

batch_size = 128
num_steps = 5000
report_every = 250
starting_learning_rate = 0.2

layer_sizes = {
  "1": 1024,
  "2": 256,
  "3": 32
}

stddevs = {
  "1": np.sqrt(2.0 / input_size) ,
  "2": np.sqrt(2.0 / layer_sizes["1"]),
  "3": np.sqrt(2.0 / layer_sizes["2"]),
  "out": np.sqrt(2.0 / layer_sizes["3"]),
}

keep_probs = {
  "1": 0.4,
  "2": 0.6,
  "3": 0.8,
}

betas = {
  "1": 0.001,
  "2": 0.001,
  "3": 0.001,
  "4": 0.001,
}

weights = {}
biases = {}
layers = {}

graph = tf.Graph()
with graph.as_default():
  
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # first hidden layer
  weights["1"] = tf.Variable(tf.truncated_normal(
    [input_size, layer_sizes["1"]], stddev=stddevs["1"]))
  biases["1"] = tf.Variable(tf.zeros([layer_sizes["1"]]))
  layers["1"] = tf.nn.dropout(
    tf.nn.relu(tf.matmul(tf_train_dataset, weights["1"]) + biases["1"]),
    keep_probs["1"])
  
  # second hidden layer
  weights["2"] = tf.Variable(tf.truncated_normal(
    [layer_sizes["1"], layer_sizes["2"]], stddev=stddevs["2"]))
  biases["2"] = tf.Variable(tf.zeros([layer_sizes["2"]]))
  layers["2"] = tf.nn.dropout(
    tf.nn.relu(tf.matmul(layers["1"], weights["2"]) + biases["2"]),
    keep_probs["2"])
  
  # third hidden layer
  weights["3"] = tf.Variable(tf.truncated_normal(
    [layer_sizes["2"], layer_sizes["3"]], stddev=stddevs["3"]))
  biases["3"] = tf.Variable(tf.zeros([layer_sizes["3"]]))
  layers["3"] = tf.nn.dropout(
    tf.nn.relu(tf.matmul(layers["2"], weights["3"]) + biases["3"]), 
    keep_probs["3"])
  
  # output layer
  weights["out"] = tf.Variable(tf.truncated_normal(
    [layer_sizes["3"], num_labels], stddev=stddevs["out"]))
  biases["out"] = tf.Variable(tf.zeros([num_labels]))
  
  # logit layer
  logits = tf.matmul(layers["3"], weights["out"]) + biases["out"]

  # calculate the loss with regularization
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels))
  loss += (betas["1"] * tf.nn.l2_loss(weights["1"]) +
           betas["2"] * tf.nn.l2_loss(weights["2"]) +
           betas["3"] * tf.nn.l2_loss(weights["3"]) +
           betas["4"] * tf.nn.l2_loss(weights["out"]))
  
  # learn with exponential rate decay.
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 100000, 0.96, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # train prediction
  train_prediction = tf.nn.softmax(logits)

  # setup validation prediction step.
  validation_layers = {}
  validation_layers["1"] = tf.nn.relu(tf.matmul(tf_valid_dataset, weights["1"]) + biases["1"])
  validation_layers["2"] = tf.nn.relu(tf.matmul(validation_layers["1"], weights["2"]) + biases["2"])
  validation_layers["3"] = tf.nn.relu(tf.matmul(validation_layers["2"], weights["3"]) + biases["3"])
  validation_logits = tf.matmul(validation_layers["3"], weights["out"]) + biases["out"]
  validation_prediction = tf.nn.softmax(validation_logits)

  # and setup the test prediction step.  
  test_layers = {}
  test_layers["1"] = tf.nn.relu(tf.matmul(tf_test_dataset, weights["1"]) + biases["1"])
  test_layers["2"] = tf.nn.relu(tf.matmul(test_layers["1"], weights["2"]) + biases["2"])
  test_layers["3"] = tf.nn.relu(tf.matmul(test_layers["2"], weights["3"]) + biases["3"])
  test_logits = tf.matmul(test_layers["3"], weights["out"]) + biases["out"]
  test_prediction = tf.nn.softmax(test_logits)
  
  saver = tf.train.Saver()

accuracy_over_time = []
steps = []
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized\n")

  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    accuracy_over_time.append(accuracy(predictions, batch_labels))
    steps.append(step)
    
    if (step % report_every == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%\n" % accuracy(validation_prediction.eval(), valid_labels))
  
  print("  Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  steps, accuracy_over_time = accuracy_averaged(steps, accuracy_over_time, 40)
  # plt.plot(steps, accuracy_over_time)
  # plt.xlabel("steps")
  # plt.ylabel("accuracy")
  # plt.show()

  # Save the final model
  model_folder = trained_models_folder + model_name
  makedir(model_folder)
  saver.save(session, model_folder + "/" + model_name)