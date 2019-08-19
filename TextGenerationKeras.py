from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time


##########################################       NESTO STEKA S GPU, ali skripta radi    ############################################

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#comment or uncomment this line to switch off GPU
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#exit()


##### LOAD IN TEXT DATA ####
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print ('Length of text: {} characters'.format(len(text)))
print(text[:250])
# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))


#### VECTORIZE THE TEXT ####
char2idx = {u:i for i, u in enumerate(vocab)}#dict, k=char, v=idx
idx2char = np.array(vocab)#list-like vector of chars, idx are actual list indexes (e.g. D which has an index 16 is on the 16th place of the vec)
text_as_int = np.array([char2idx[c] for c in text])#vektor stvoren od liste svih V dobivenih putem char2idx za sve char u text
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


#### PREPARING THE DATASET ####
seq_length = 100
examples_per_epoch = len(text)//seq_length # // is division without remainder
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for i in char_dataset.take(5):
  print(idx2char[i.numpy()])
#.take(N) je metoda iz tf da uzme prvih N. #i.numpy() je i kao indeks za numpy vektor
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)#what does batch() do?
#for item in sequences.take(5):
#  print(repr(''.join(idx2char[item.numpy()])))
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)# this uses split_input_target to create two int encoded strings of chars, the first is "Firs" and the second is "irst", for example. It does this for a string "First" named chunk... if you pass in a longer string it will do for that, or a shorter string for that also.
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
#Each index of these vectors are processed as one time step. For the input at time step 0, the model receives the index for "F" and trys to predict the index for "i" as the next character. At the next timestep, it does the same thing but the RNN considers the previous step context in addition to the current input character.
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
# Create batches #
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
#print(dataset)




#### BUILDING THE MODEL ####

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024 # Number of RNN units
#use GPU if available
if tf.test.is_gpu_available():
	print("Computing on GPU")
	#rnn = tf.keras.layers.CuDNNGRU
#else:
import functools
rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),rnn(rnn_units,return_sequences=True, recurrent_initializer='glorot_uniform',stateful=True),tf.keras.layers.Dense(vocab_size)])
	return model
model = build_model(vocab_size = len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)
#For each character the model looks up the embedding, runs the GRU one timestep with the embedding as input, and applies the dense layer to generate logits predicting the log-liklihood of the next character



#### TESTING THE MODEL SHAPES AND FLOW ####
for input_example_batch, target_example_batch in dataset.take(1): 
	example_batch_predictions = model(input_example_batch)
	print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
print(model.summary())
# It is important to sample from this distribution as taking the argmax of the distribution can easily get the model stuck in a loop.
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
#let us see what we got...
print(sampled_indices)
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions (before training the RNN): \n", repr("".join(idx2char[sampled_indices ]))) 




####   TRAINING   ####
#At this point the problem can be treated as a standard classification problem. Given the previous RNN state, and the input this time step, predict the class of the next character.
def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())
model.compile(optimizer = tf.train.AdamOptimizer(),loss = loss)



from pathlib import Path
home = str(Path.home())
## Directory where the checkpoints will be saved
checkpoint_dir = home
## Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)



#### UNCOMMENT THE FOLLOWING TO TRAIN!!!!!! #####
EPOCHS=1
history = model.fit(dataset.repeat().make_one_shot_iterator(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
###########################################

###   GENERATE TEXT   ###

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()



def generate_text(model, start_string):
	# Evaluation step (generating text using the learned model)
	# Number of characters to generate
	num_generate = 1000
	# Converting our start string to numbers (vectorizing) 
	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval, 0)
	# Empty string to store our results
	text_generated = []
	# Low temperatures results in more predictable text.
	# Higher temperatures results in more surprising text.
	# Experiment to find the best setting.
	temperature = 1.0
	# Here batch size == 1
	model.reset_states()
	for i in range(num_generate):
		predictions = model(input_eval)
		# remove the batch dimension
		predictions = tf.squeeze(predictions, 0)
		# using a multinomial distribution to predict the word returned by the model
		predictions = predictions / temperature
		predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
		# We pass the predicted word as the next input to the model
		# along with the previous hidden state
		input_eval = tf.expand_dims([predicted_id], 0)
		text_generated.append(idx2char[predicted_id])
	return (start_string + ''.join(text_generated))

gtext = generate_text(model, start_string=u"ROMEO: ")
print(gtext)

dest_file=open("generated.txt","w")
dest_file.writelines(gtext)
dest_file.close()



