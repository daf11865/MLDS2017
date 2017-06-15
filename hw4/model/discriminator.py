import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import sys
sys.path.append('utils')
import pprint

def randTrue(prob):
	return np.random.binomial(1,prob)

class D(object):
	def __init__(self, config, pretrain_ebd_param):
		self.config = config
		self.build_model(pretrain_ebd_param)

	def build_model(self, pretrain_ebd_param):
		with tf.variable_scope('D'): 
			batch_size = self.config['batch_size']
			cell_size = self.config['cell_size']
			lr_init = self.config['lr_init']
			lr_decay_step = self.config['lr_decay_step']
			lr_decay = self.config['lr_decay']
			lr_min = self.config['lr_min']
			ebd_trainable = self.config['ebd_trainable']

			#cell = tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=1.0, state_is_tuple=True)
			cell_size /= 2 
			cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=1.0, state_is_tuple=True)]*2, state_is_tuple=True)

			self.x = tf.placeholder(tf.int32, [None, None])
			self.x_seq_len = tf.placeholder(tf.int32, [None])

			w_ebd = tf.get_variable('w_ebd', initializer=tf.constant_initializer(pretrain_ebd_param), shape=pretrain_ebd_param.shape, trainable=ebd_trainable)
			x_ebd = tf.nn.embedding_lookup(w_ebd, self.x)

			enc_out, enc_state = tf.nn.dynamic_rnn(cell, x_ebd, sequence_length=self.x_seq_len, scope='enc', dtype=tf.float32) # 

			self.y = tf.placeholder(tf.int32, [None, None])
			self.y_seq_len = tf.placeholder(tf.int32, [None])

			y_ebd = tf.nn.embedding_lookup(w_ebd, self.y)

			attn_keys, attn_values, attn_score_fn, attn_construct_fn = tf.contrib.seq2seq.prepare_attention(enc_out, 'bahdanau', cell_size)
			dec_fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(enc_state, attn_keys, attn_values, attn_score_fn, attn_construct_fn)
			dec_out, dec_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell, dec_fn_train, inputs=y_ebd, sequence_length=self.y_seq_len, scope='dec')

			'''max_time = tf.shape(dec_out)[1]
			enc_out_tile = tf.expand_dims(enc_state[1],1)
			enc_out_tile = tf.tile(enc_out_tile,[1,max_time,1])
			concat = tf.concat([enc_out_tile,dec_out], -1) #[batch_size, time_step, cell_size]

			self.logits = tcl.fully_connected(concat, 1, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False), activation_fn=None, scope='logits')[:,:,0]'''
			self.logits = tcl.fully_connected(dec_out, 1, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False), activation_fn=None, scope='logits')[:,:,0]
			self.reward = tf.sigmoid(self.logits)*2 - 1.

			self.label =  tf.placeholder(tf.float32, [None, None])
			self.loss_weight = tf.placeholder(tf.float32, [None, None])
			self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits) * self.loss_weight) / tf.reduce_sum(self.loss_weight)

			self.step = tf.Variable(0, name='d_step', dtype=tf.int32, trainable=False)
			self.lr = tf.maximum(tf.train.exponential_decay(lr_init, self.step, lr_decay_step, lr_decay, staircase=True),lr_min)
			#self.opt = tf.train.MomentumOptimizer(self.lr, 0.5).minimize(self.loss, global_step = self.step)
			self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step = self.step)


	def step_GAN(self, sess, x, y, label):
		x_seq_len = [sent.index(2) for sent in x]
		y_seq_len = [sent.index(2)+1 if 2 in sent else len(sent) for sent in y] # w2i['<EOS>'] = 2
		#loss_weight = [[1. if t<length and randTrue(float(t+1)/length) else 0. for t in range(max(y_seq_len))] for length in y_seq_len]
		loss_weight = [[1. if t<length else 0. for t in range(max(y_seq_len))] for length in y_seq_len]
		fetch = [self.opt, 
						self.loss,
						self.step,
						self.lr]
		feed = {self.x:x,
						self.x_seq_len:x_seq_len,
						self.y:y,
						self.y_seq_len:y_seq_len,
						self.label:label,
						self.loss_weight:loss_weight}
		fetch_ = sess.run(fetch, feed)

		return fetch_

	def get_reward(self, sess, x, y):
		x_seq_len = [sent.index(2) for sent in x] # w2i['<EOS>'] = 2
		y_seq_len = [sent.index(2)+1 if 2 in sent else len(sent) for sent in y] # w2i['<EOS>'] = 2

		fetch = [self.reward,self.logits]
		feed = {self.x:x,
						self.x_seq_len:x_seq_len,
						self.y:y,
						self.y_seq_len:y_seq_len}
		reward,logits = sess.run(fetch, feed)

		for i in range(len(y_seq_len)):
			reward[i,y_seq_len[i]:] = 0.
		return reward

