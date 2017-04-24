import tensorflow as tf
import math
import beam_search
import numpy as np
import sys
sys.path.append('utils')
import rnn

"""
Dr1 : c_dec_ebd, c_dec_out(including hidden state)
Dr2 : v_enc_ebd, v_enc_out(excluding hidden state)(as I'm lazy to modify code, should this one not dropout due to not syncornize v_enc_out?)
video bilstm has same parametera as caption unilstm
"""

class Model(object):

	def __init__(self):
		self.hid_size = 300
		self.v_cell = [tf.contrib.rnn.BasicLSTMCell(self.hid_size/2, forget_bias=1.0, state_is_tuple=True),
									tf.contrib.rnn.LayerNormBasicLSTMCell(self.hid_size/2, forget_bias=1.0, layer_norm=True)][0]
		self.c_cell = [tf.contrib.rnn.BasicLSTMCell(self.hid_size, forget_bias=1.0, state_is_tuple=True),
									tf.contrib.rnn.LayerNormBasicLSTMCell(self.hid_size, forget_bias=1.0, layer_norm=True)][0]	
		self.dec_step = 40
		self.tie_ebd = False

	def get_config_str(self):
		return str("BiPyramid_Dr1_te{}_hs{}".format(self.tie_ebd, self.hid_size))


	# In order to implement more independent components, pad training data (caption input, label, and loss mask) to have known sequence length.
	def train_data_second_process(self, c_idx_padded, l_idx_padded, loss_mask):
		max_seq_len = len(c_idx_padded[0,:self.dec_step])
		c_idx_padded = np.lib.pad(c_idx_padded[:,:self.dec_step], [[0,0], [0,self.dec_step-max_seq_len]], 'constant').astype(np.int32)
		l_idx_padded = np.lib.pad(l_idx_padded[:,:self.dec_step], [[0,0], [0,self.dec_step-max_seq_len]], 'constant').astype(np.int32)
		loss_mask = np.lib.pad(loss_mask[:,:self.dec_step], [[0,0], [0,self.dec_step-max_seq_len]], 'constant').astype(np.float32)
		return c_idx_padded, l_idx_padded, loss_mask


	def build_model_train(self, V_feat, C_idx, Embedding, Sd_rate):
		batch_size = tf.shape(V_feat)[0]
		voc_size = int(Embedding.get_shape()[0])
		ebd_size = int(Embedding.get_shape()[1])

		# Encoding
		with tf.variable_scope("V_LSTM") as scope:
			self.w_v_ebd = tf.get_variable("w_v_ebd", initializer=tf.truncated_normal_initializer(stddev=2/4096), shape=[4096,ebd_size])
			self.b_v_ebd = tf.get_variable("b_v_ebd", initializer=tf.constant_initializer(1e-4), shape=[ebd_size])
			v_enc_ebd = tf.reshape(tf.nn.xw_plus_b(tf.reshape(V_feat,[-1,4096]), self.w_v_ebd, self.b_v_ebd), [-1,80,ebd_size])

			v_enc_output_1, v_enc_state_1 =rnn.bidirectional_dynamic_rnn(self.v_cell, self.v_cell, v_enc_ebd, sequence_length=None, dtype=tf.float32)
			scope.reuse_variables()
			v_enc_in_2 = tf.transpose(tf.gather(tf.transpose(tf.concat(v_enc_output_1, 2), [1,0,2]), [2*t+1 for t in range(40)]), [1,0,2])
			v_enc_output_2, v_enc_state_2 = rnn.bidirectional_dynamic_rnn(self.v_cell, self.v_cell, v_enc_in_2, dtype=tf.float32)
			v_enc_in_3 = tf.transpose(tf.gather(tf.transpose(tf.concat(v_enc_output_2, 2), [1,0,2]), [2*t+1 for t in range(20)]), [1,0,2])
			v_enc_output_3, v_enc_state_3 = rnn.bidirectional_dynamic_rnn(self.v_cell, self.v_cell, v_enc_in_3, dtype=tf.float32)
			v_enc_in_4 = tf.transpose(tf.gather(tf.transpose(tf.concat(v_enc_output_3, 2), [1,0,2]), [2*t+1 for t in range(10)]), [1,0,2])
			v_enc_output_4, v_enc_state_4 = rnn.bidirectional_dynamic_rnn(self.v_cell, self.v_cell, v_enc_in_4, dtype=tf.float32)

			v_enc_fw , v_enc_bw = v_enc_output_4
			v_enc_output = tf.concat([v_enc_fw[:,-1,:] , v_enc_bw[:,0,:]], 1) # [64,512]

		# Decoding
		self.w_proj = tf.transpose(Embedding) if self.tie_ebd else tf.get_variable(name="w_proj", initializer=tf.transpose(Embedding.initialized_value()))
		self.b_proj = tf.zeros([voc_size]) if self.tie_ebd else tf.get_variable(name='b_proj',initializer=tf.truncated_normal_initializer(stddev=1e-4), shape=[voc_size])
		with tf.variable_scope("C_LSTM") as scope:
			self.w_out = tf.get_variable("w_out", initializer=tf.truncated_normal_initializer(stddev=1e-4), shape=[self.hid_size*2,ebd_size])

			c_zero_state = self.c_cell.zero_state(batch_size, dtype = tf.float32)
			c_dec_state = (c_zero_state[0], tf.concat([c_zero_state[1]]*2, axis = 1))
			c_dec_idx = C_idx[:,0]
			for i in range(self.dec_step):
				c_dec_ebd = tf.nn.embedding_lookup(Embedding, c_dec_idx) # [64, 300]
				c_dec_ebd = tf.layers.dropout(c_dec_ebd, rate = 0.5, training = True)
				c_dec_output, c_dec_state = self.c_cell(c_dec_ebd, c_dec_state) # [64, 512], ([64,512],[64,512])
				scope.reuse_variables()

				concat = tf.concat([c_dec_output, v_enc_output],axis = 1) # [64,1024]
				concat = tf.layers.dropout(concat, rate = 0.5, training = True)
				out_ebd = tf.matmul(concat, self.w_out)
				logits = tf.nn.xw_plus_b(out_ebd, self.w_proj, self.b_proj)
				c_dec_idx_pred = tf.cast(tf.reshape(tf.arg_max(logits, 1), [-1]), tf.int32)

				if i < self.dec_step-1:
					toss_outcome = tf.random_uniform([batch_size], minval=0, maxval=1.0)
					c_dec_idx =  tf.where(tf.greater_equal(toss_outcome, 1-Sd_rate),  C_idx[:,i+1], c_dec_idx_pred)
					c_dec_state = (c_dec_state[0], concat)

				if i == 0:
					out_ebd_collection = tf.expand_dims(out_ebd, axis = 0) # [1,64,300]
				else:
					out_ebd_collection = tf.concat([out_ebd_collection, tf.expand_dims(out_ebd, axis = 0)], axis = 0) ## [i,64,300]

			out_ebd_collection = tf.reshape(tf.transpose(out_ebd_collection, perm = [1,0,2]), [-1,ebd_size])

		logits = tf.nn.xw_plus_b(out_ebd_collection, self.w_proj, self.b_proj)

		# For sampled loss
		ft_for_proj = out_ebd_collection
		w_proj_t = tf.transpose(self.w_proj)
		b_proj = self.b_proj
		logits = logits

		return ft_for_proj, w_proj_t, b_proj, logits


	def build_model_inference(self, V_feat, Embedding, beam_width = 5):
		voc_size = int(Embedding.get_shape()[0])
		ebd_size = int(Embedding.get_shape()[1])

		# Encoding
		with tf.variable_scope("V_LSTM",reuse = True):
			v_enc_ebd = tf.reshape(tf.nn.xw_plus_b(tf.reshape(V_feat,[-1,4096]), self.w_v_ebd, self.b_v_ebd), [-1,80,ebd_size])

			v_enc_output_1, v_enc_state_1 = rnn.bidirectional_dynamic_rnn(self.v_cell, self.v_cell, v_enc_ebd, sequence_length=None, dtype=tf.float32)
			v_enc_in_2 = tf.transpose(tf.gather(tf.transpose(tf.concat(v_enc_output_1, 2), [1,0,2]), [2*t+1 for t in range(40)]), [1,0,2])
			v_enc_output_2, v_enc_state_2 = rnn.bidirectional_dynamic_rnn(self.v_cell, self.v_cell, v_enc_in_2, dtype=tf.float32)
			v_enc_in_3 = tf.transpose(tf.gather(tf.transpose(tf.concat(v_enc_output_2, 2), [1,0,2]), [2*t+1 for t in range(20)]), [1,0,2])
			v_enc_output_3, v_enc_state_3 = rnn.bidirectional_dynamic_rnn(self.v_cell, self.v_cell, v_enc_in_3, dtype=tf.float32)
			v_enc_in_4 = tf.transpose(tf.gather(tf.transpose(tf.concat(v_enc_output_3, 2), [1,0,2]), [2*t+1 for t in range(10)]), [1,0,2])
			v_enc_output_4, v_enc_state_4 = rnn.bidirectional_dynamic_rnn(self.v_cell, self.v_cell, v_enc_in_4, dtype=tf.float32)

			v_enc_fw , v_enc_bw = v_enc_output_4
			v_enc_output = tf.concat([v_enc_fw[:,-1,:] , v_enc_bw[:,0,:]], 1) # [64,512]

		# Decoding
		bs = beam_search.Beam_Search(beam_width = beam_width)

		c_dec_idx = tf.fill([1], 1)
		c_zero_state = self.c_cell.zero_state(1, dtype = tf.float32)
		c_dec_state = (c_zero_state[0], tf.concat([c_zero_state[1]]*2, axis = 1))
		for i in range(self.dec_step):
			with tf.variable_scope("C_LSTM",reuse = True):
				c_dec_ebd = tf.nn.embedding_lookup(Embedding, c_dec_idx)
				c_dec_output, c_dec_state = self.c_cell(c_dec_ebd, c_dec_state)

				concat = tf.concat([c_dec_output, v_enc_output],axis = 1)
				c_dec_state = (c_dec_state[0], concat)
				out_ebd = tf.matmul(concat, self.w_out)

			logits = tf.nn.xw_plus_b(out_ebd, self.w_proj, self.b_proj)
			probs = tf.nn.softmax(logits)

			c_dec_idx, c_dec_state, beam_gen, beam_score = bs.step(probs, c_dec_state)
			if i == 0:
				v_enc_output = tf.concat([v_enc_output]*beam_width,axis = 0)

		return beam_gen, beam_score
