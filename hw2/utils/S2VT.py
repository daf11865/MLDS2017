import tensorflow as tf
import math
import beam_search
import random
import numpy as np

"""
Dr1':  c_dec_ebd, c_dec_out(excluding hidden state)
"""

class Model(object):

	def __init__(self):
		self.hid_size = 256
		self.v_cell = self.c_cell = [tf.contrib.rnn.BasicLSTMCell(self.hid_size, forget_bias=1.0, state_is_tuple=True),
									tf.contrib.rnn.LayerNormBasicLSTMCell(self.hid_size, forget_bias=1.0, layer_norm=True)][1]
		self.dec_step = 40

	def get_config_str(self):
		return str("S2VT_Shared_LN_Dr1'_hs{}".format(self.hid_size))

	def train_data_second_process(self,c_idx_padded, l_idx_padded, loss_mask):
		return c_idx_padded, l_idx_padded, loss_mask

	def build_model_train(self, V_feat, C_idx, Embedding, Sd_rate):
		batch_size = tf.shape(V_feat)[0]
		v_seq_len = tf.shape(V_feat)[1]
		c_seq_len = tf.shape(C_idx)[1]
		voc_size = int(Embedding.get_shape()[0])
		ebd_size = int(Embedding.get_shape()[1])

		# Encoding
		v_ebd_size = self.hid_size + ebd_size
		with tf.variable_scope("V_ebd"):
			self.w_v_ebd = tf.get_variable("w", initializer=tf.truncated_normal_initializer(stddev=1e-4), shape=[4096,v_ebd_size])
			self.b_v_ebd = tf.get_variable("b", initializer=tf.constant_initializer(1e-4), shape=[v_ebd_size])
			v_feat_flat = tf.reshape(V_feat,[-1,4096])
			v_enc_ebd = tf.reshape(tf.nn.xw_plus_b(v_feat_flat, self.w_v_ebd, self.b_v_ebd), [-1,v_seq_len,v_ebd_size])

		with tf.variable_scope("LSTM") as scope:
			v_enc_output, v_enc_state = tf.nn.dynamic_rnn(cell = self.v_cell, 
																										dtype = tf.float32, 
																										inputs = tf.layers.dropout(v_enc_ebd, rate = 0.5, training = True),
																										scope = "enc_dec")
			scope.reuse_variables()
			padding = tf.fill([batch_size, v_seq_len, ebd_size], 0.0)
			c_enc_output, c_enc_state  = tf.nn.dynamic_rnn(cell = self.c_cell, 
																										dtype = tf.float32, 
																										inputs = tf.layers.dropout(tf.concat([padding,v_enc_output],axis=2), rate = 0.5, training = True),
																										scope = "enc_dec")
		# Decoding - training mode
		with tf.variable_scope("LSTM", reuse = True) as scope:
			padding = tf.fill([batch_size, c_seq_len, v_ebd_size], 0.0)
			v_dec_output, v_dec_state = tf.nn.dynamic_rnn(cell = self.v_cell, 
																										dtype = tf.float32, 
																										inputs = padding,
																										initial_state = v_enc_state,
																										scope = "enc_dec")
			c_dec_ebd = tf.nn.embedding_lookup(Embedding, C_idx)
			c_dec_output, c_dec_state = tf.nn.dynamic_rnn(cell = self.c_cell, 
																										dtype = tf.float32, 
																										inputs =  tf.layers.dropout(tf.concat([c_dec_ebd,v_dec_output], axis = 2), rate = 0.5, training = True),
																										initial_state = c_enc_state,
																										scope = "enc_dec")

			c_dec_output = tf.layers.dropout(c_dec_output, rate = 0.5, training = True)

		self.w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1e-4), shape=[self.hid_size, ebd_size])
		self.b_out = tf.get_variable(name='b_out', initializer=tf.truncated_normal_initializer(stddev=1e-4), shape=[ebd_size])
		out_ebd = tf.nn.xw_plus_b(tf.reshape(c_dec_output, [-1,self.hid_size]), self.w_out, self.b_out)

		self.w_proj = tf.get_variable(name="w_proj", initializer=tf.transpose(Embedding.initialized_value()))
		self.b_proj = tf.get_variable(name='b_proj', initializer=tf.truncated_normal_initializer(stddev=1e-4), shape=[voc_size])
		logits = tf.nn.xw_plus_b(out_ebd, self.w_proj, self.b_proj)

		ft_for_proj = out_ebd
		w_proj_t = tf.transpose(self.w_proj)
		b_proj = self.b_proj
		logits = logits
		return ft_for_proj, w_proj_t, b_proj, logits

	def build_model_inference(self, V_feat, Embedding, beam_width = 5):
		batch_size = tf.shape(V_feat)[0]
		v_seq_len = tf.shape(V_feat)[1]
		voc_size = int(Embedding.get_shape()[0])
		ebd_size = int(Embedding.get_shape()[1])

		# Encoding
		v_ebd_size = self.hid_size + ebd_size
		with tf.variable_scope("V_ebd", reuse = True):
			v_feat_flat = tf.reshape(V_feat,[-1,4096])
			v_enc_ebd = tf.reshape(tf.nn.xw_plus_b(v_feat_flat, self.w_v_ebd, self.b_v_ebd), [-1,v_seq_len,v_ebd_size])

		with tf.variable_scope("LSTM", reuse = True):
			v_enc_output, v_enc_state = tf.nn.dynamic_rnn(cell = self.v_cell, 
																										dtype = tf.float32, 
																										inputs = v_enc_ebd,
																										scope = "enc_dec")
			padding = tf.fill([batch_size, v_seq_len, ebd_size], 0.0)
			c_enc_output, c_enc_state  = tf.nn.dynamic_rnn(cell = self.c_cell, 
																										dtype = tf.float32, 
																										inputs = tf.concat([padding,v_enc_output],axis=2),
																										scope = "enc_dec")

		# Decoding - inference with beam search
		bs = beam_search.Beam_Search(beam_width = beam_width)

		v_dec_padding = tf.fill([1,v_ebd_size], 0.0)
		c_dec_ebd = tf.nn.embedding_lookup(Embedding, tf.fill([1], 1))
		for i in range(self.dec_step):
			if i == 0:
				with tf.variable_scope("LSTM/enc_dec",reuse=True):
					v_dec_output, v_dec_state = self.v_cell(v_dec_padding, v_enc_state)
					c_dec_output, c_dec_state = self.c_cell(tf.concat([c_dec_ebd, v_dec_output], axis=1), c_enc_state)

				out_ebd = tf.nn.xw_plus_b(c_dec_output, self.w_out, self.b_out)
				logits = tf.nn.xw_plus_b(out_ebd, self.w_proj, self.b_proj)
				probs = tf.nn.softmax(logits)

				# Beam search
				c_dec_idx, c_dec_state, beam_gen, beam_score = bs.step(probs, c_dec_state)
				c_dec_ebd = tf.nn.embedding_lookup(Embedding, c_dec_idx)

				v_dec_state = (tf.concat([v_dec_state[0]]*beam_width, axis = 0), tf.concat([v_dec_state[1]]*beam_width, axis = 0))
				v_dec_padding = tf.fill([beam_width,v_ebd_size], 0.0)

			else:
				with tf.variable_scope("LSTM/enc_dec",reuse=True):
					v_dec_output, v_dec_state = self.v_cell(v_dec_padding, v_dec_state)
					concat_ebd = tf.concat([c_dec_ebd, v_dec_output], axis=1)
					c_dec_output, c_dec_state = self.c_cell(concat_ebd, c_dec_state)


				out_ebd = tf.nn.xw_plus_b(c_dec_output, self.w_out, self.b_out)
				logits = tf.nn.xw_plus_b(out_ebd, self.w_proj, self.b_proj)
				probs = tf.nn.softmax(logits)

				# Beam search
				c_dec_idx, c_dec_state, beam_gen, beam_score = bs.step(probs, c_dec_state)
				c_dec_ebd = tf.nn.embedding_lookup(Embedding, c_dec_idx)

		return beam_gen, beam_score
