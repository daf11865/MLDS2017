import tensorflow as tf
import math

def pred_discr_BiRNN_Fc(X_id,
									Seq_len,
									Drop_keep,
									voc_size,
									embedding_size,
									hidden_size,
									num_stack_layer,
									basic_cell,
									embedding,
									Candidate_id):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
	stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

	#with tf.variable_scope("predictor"):
	# Bidirectional
	with tf.variable_scope("BiRNN"):
		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out_pred = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_pred_flat = tf.reshape(cell_out_pred, [-1,2*hidden_size])

	# Fully connected
	with tf.variable_scope("Fc"):
		w_pred = tf.get_variable(name="w_",initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(2*hidden_size)),shape=[2*hidden_size, 2*hidden_size])
		b_pred = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[2*hidden_size])
		fc_pred = tf.nn.dropout(tf.matmul(cell_out_pred_flat, w_pred) + b_pred, Drop_keep)

	# Output predict layer
	w_pred_out = tf.get_variable(name="w_out",initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(2*hidden_size)),shape=[2*hidden_size, voc_size])
	b_pred_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	pred_logits = tf.matmul(fc_pred, w_pred_out) + b_pred_out
	pred_probs = tf.nn.softmax(pred_logits)	

	w_pred_out_t = tf.transpose(w_pred_out)

	with tf.variable_scope("discriminator"):
		with tf.variable_scope("BiRNN"):
			[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																					cell_bw=stack_cell,
																																					dtype=tf.float32,
																																					sequence_length=Seq_len,
																																					inputs=X_ebd)
			cell_out_sentence = tf.concat([outputs_fw[:,-1], outputs_bw[:,0]],1) # [batch_size, 2*hidden_num]
			cell_out_candidate = tf.concat([tf.gather_nd(outputs_fw,Candidate_id),tf.gather_nd(outputs_bw,Candidate_id)],axis=1) # [batch_size, 2*hidden_num]
			cell_out_attn = tf.concat([cell_out_sentence,cell_out_candidate],1) # [batch_size, 4*hidden_num]
			
			cell_out = cell_out_candidate # Stay tuned
		# Fully connected
		with tf.variable_scope("Fc"):
			w_discr = tf.get_variable(name="w_",initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(4*hidden_size)),shape=[2*hidden_size, 2*hidden_size])
			b_discr = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[2*hidden_size])
			fc_discr = tf.nn.dropout(tf.matmul(cell_out, w_discr) + b_discr, Drop_keep)

		# Output discriminate layer
		w_discr_out = tf.get_variable(name="w_out",initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(2*hidden_size)),shape=[2*hidden_size, 1])
		discr_logit = tf.matmul(fc_discr, w_discr_out)
		discr_prob = tf.sigmoid(discr_logit)	

	return [fc_pred, w_pred_out_t, b_pred_out, pred_logits, pred_probs], [discr_logit, discr_prob]

def RNN_model(RNN_type, kwargs):
	if RNN_type == 'Pred_Discr_BiRNN_Fc':
		[ft_before_out, w_out_t, b_out, out1, out2], [discr_logit, discr_prob] = pred_discr_BiRNN_Fc(**kwargs)

	return [ft_before_out, w_out_t, b_out, out1, out2], [discr_logit, discr_prob]

