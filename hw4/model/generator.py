import tensorflow as tf
import tensorflow.contrib.layers as tcl
import tensorflow.contrib as tc
import numpy as np
import sys
sys.path.append('utils')
import seq2seq
import pprint

def randTrue(prob):
	return np.random.binomial(1,prob)

class G(object):
	def __init__(self, config, pretrain_ebd_param):
		self.config = config
		self.build_model(pretrain_ebd_param)

	def build_model(self, pretrain_ebd_param):
		with tf.variable_scope('G') as scope: 
			batch_size = self.config['batch_size']
			cell_size = self.config['cell_size']
			ebd_size = self.config['ebd_size']
			voc_size = self.config['voc_size']
			lr_init = self.config['lr_init']
			lr_decay_step = self.config['lr_decay_step']
			lr_decay = self.config['lr_decay']
			lr_min = self.config['lr_min']
			max_time = self.config['max_time_step']
			ebd_trainable = self.config['ebd_trainable']

			cell = tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=1.0, state_is_tuple=True)
			#cell_multi = tf.contrib.rnn.MultiRNNCell([cell]*2, state_is_tuple=True)

			self.x = tf.placeholder(tf.int32, [None, None])
			self.x_seq_len = tf.placeholder(tf.int32, [None])

			w_ebd = tf.get_variable('w_ebd', initializer=tf.constant_initializer(pretrain_ebd_param), shape=pretrain_ebd_param.shape, trainable=ebd_trainable)
			x_ebd = tf.nn.embedding_lookup(w_ebd, self.x)

			enc_out, enc_state = tf.nn.dynamic_rnn(cell, x_ebd, sequence_length=self.x_seq_len, scope='enc', dtype=tf.float32)

			self.y = tf.placeholder(tf.int32, [None, None])
			self.y_seq_len = tf.placeholder(tf.int32, [None])
			self.loss_weight = tf.placeholder(tf.float32, [None, None])
			self.reward = tf.placeholder(tf.float32, [None, None])

			y_ebd = tf.nn.embedding_lookup(w_ebd, self.y)

			attn_keys, attn_values, attn_score_fn, attn_construct_fn = tc.seq2seq.prepare_attention(enc_out, 'bahdanau', cell_size)
			
			dec_fn_train = tc.seq2seq.attention_decoder_fn_train(enc_state, attn_keys, attn_values, attn_score_fn, attn_construct_fn)
			dec_out_train, _, _ = tc.seq2seq.dynamic_rnn_decoder(cell, dec_fn_train, inputs=y_ebd, sequence_length=self.y_seq_len, scope='dec') # [None, timestep, cellsize]

			def out_func(cell_out):
				fc = tcl.fully_connected(cell_out, ebd_size, weights_initializer=tcl.xavier_initializer(uniform=False), scope='fc')
				lgt = tcl.fully_connected(fc, voc_size, weights_initializer=tf.constant_initializer(pretrain_ebd_param.transpose()), 
																										biases_initializer=None, activation_fn=None,scope='logits') # [None, voc_size]
				return lgt

			with tf.variable_scope('dec') as scope: 
				lgt_train = out_func(dec_out_train)
				prob_train = tf.nn.softmax(lgt_train)

			lbl = self.y[:,1:]
			lgt = lgt_train[:,:-1]
			prob = prob_train[:,:-1]
			lw = self.loss_weight[:,1:]
			r = self.reward[:,1:]
			# MLE
			softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl, logits=lgt)
			self.perplex_mle =  tf.exp( tf.reduce_sum(softmax_loss) / tf.to_float(tf.reduce_sum(self.y_seq_len)) ) #tf.reduce_sum( tf.exp( tf.reduce_sum(softmax_loss,1) / tf.to_float(self.y_seq_len) ) ) / batch_size
			self.loss_mle = tf.reduce_sum(softmax_loss * lw) / tf.to_float(tf.reduce_sum(self.y_seq_len)) #tf.to_float(tf.reduce_sum(self.y_seq_len))
			self.step_mle = tf.Variable(0, name='g_step_mle', dtype=tf.int32, trainable=False)
			self.lr_mle = tf.maximum(tf.train.exponential_decay(lr_init, self.step_mle, lr_decay_step, lr_decay, staircase=True),lr_min)
			self.opt_mle = tf.train.AdamOptimizer(self.lr_mle).minimize(self.loss_mle, global_step = self.step_mle)

			# GAN
			lbl_multivar = tf.one_hot(lbl, voc_size, 1, 0, dtype=tf.int32) # [batch_size, time_step, voc_size]
			lbl_indice = tf.where(tf.equal(lbl_multivar, 1)) # [batch_size * time_step]
			prob_gather = tf.reshape(tf.gather_nd(prob_train, lbl_indice), [batch_size, -1] )# [batch_size , time_step]
			prob_prime = tf.clip_by_value( tf.where(r>0.,prob_gather, 1-prob_gather), 1e-10, 1-1e-10 )
			self.loss_gan = -tf.reduce_sum(tf.log(prob_prime) * tf.abs(r)) / tf.to_float(tf.reduce_sum(self.y_seq_len)) #tf.to_float(batch_size)
			#self.loss_gan = -tf.reduce_sum(tf.log(prob_prime) * tf.abs(r)) / tf.reduce_sum(self.y_seq_len) / batch_size
			self.step_gan = tf.Variable(0, name='g_step_gan', dtype=tf.int32, trainable=False)
			self.lr_gan = tf.maximum(tf.train.exponential_decay(lr_init, self.step_gan, lr_decay_step, lr_decay, staircase=True),lr_min)
			self.opt_gan = tf.train.AdamOptimizer(self.lr_gan).minimize(self.loss_gan, global_step = self.step_gan)

			# Monte Carlo inference
			with tf.variable_scope(tf.get_variable_scope(), reuse=True): 
				dec_fn_inference = seq2seq.attn_monte_carlo_dec_fn_inference(out_func, enc_state, attn_keys, attn_values, attn_score_fn, attn_construct_fn, w_ebd, 1, 2, max_time, voc_size)
				dec_out_inference, _, _ = tc.seq2seq.dynamic_rnn_decoder(cell, dec_fn_inference, scope='dec') # [None, timestep, voc_size]
				self.pred_mc_idx = tf.argmax(dec_out_inference,-1)#tf.where(tf.greater(dec_out_inference,0.))
			# Arg max inference
			with tf.variable_scope(tf.get_variable_scope(), reuse=True): 
				dec_fn_inference = tc.seq2seq.attention_decoder_fn_inference(out_func, enc_state, attn_keys, attn_values, attn_score_fn, attn_construct_fn, w_ebd, 1, 2, max_time, voc_size)
				dec_out_inference, _, _ = tc.seq2seq.dynamic_rnn_decoder(cell, dec_fn_inference, scope='dec') # [None, timestep, voc_size]
				self.pred_idx = tf.argmax(dec_out_inference,-1)

			self.lbl_multivar = lbl_multivar
			self.lbl_indice = lbl_indice
			self.prob_gather = prob_gather
			self.prob_prime = prob_prime

	def step_MLE(self, sess, x, y, w):
		x_seq_len = [sent.index(2) for sent in x]
		y_seq_len = [sent.index(2)+1 if 2 in sent else len(sent) for sent in y]

		fetch = [self.opt_mle, 
						self.perplex_mle,
						self.step_mle,
						self.lr_mle]
		feed = {self.x:x,
						self.x_seq_len:x_seq_len,
						self.y:y,
						self.y_seq_len:y_seq_len,
						self.loss_weight:w}
		fetch_ = sess.run(fetch, feed)

		return fetch_

	def step_GAN(self, sess, x, y, r):
		x_seq_len = [sent.index(2) for sent in x]
		y_seq_len = [sent.index(2)+1 if 2 in sent else len(sent) for sent in y] # w2i['<EOS>'] = 2
		#sampled_reward = [[r[i][t] if t<y_seq_len[i] and randTrue(float(t+1)/y_seq_len[i]) else 0. for t in range(max(y_seq_len))] for i in range(len(y_seq_len))]
		#weighted_reward = [[r[i][t]*(t+1)/y_seq_len[i] if t<y_seq_len[i] else 0. for t in range(max(y_seq_len))] for i in range(len(y_seq_len))]
		reward = [[r[i][t] if t<y_seq_len[i] else 0. for t in range(max(y_seq_len))] for i in range(len(y_seq_len))]

		fetch = [self.opt_gan, 
						self.loss_gan,
						self.step_gan,
						self.lr_gan]
		feed = {self.x:x,
						self.x_seq_len:x_seq_len,
						self.y:y,
						self.y_seq_len:y_seq_len,
						self.reward:reward}
		fetch_ = sess.run(fetch, feed_dict=feed)




		return fetch_


	def generate(self, sess, x, monte_carlo=True):
		x_seq_len = [sent.index(2) for sent in x] # w2i['<EOS>'] = 2
		feed = {self.x:x,
						self.x_seq_len:x_seq_len}
		if monte_carlo:
			fetch = [self.pred_mc_idx]
			pred_idx = sess.run(fetch, feed)[0]	
		else:
			fetch = [self.pred_idx]
			pred_idx = sess.run(fetch, feed)[0]	
		return np.hstack([np.ones([len(pred_idx),1]),pred_idx]).tolist()
