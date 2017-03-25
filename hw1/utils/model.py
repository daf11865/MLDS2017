import tensorflow as tf
import math

def probs_UniRNN(X_id,
								Seq_len,
								Drop_keep,
								voc_size,
								embedding_size,
								hidden_size,
								num_stack_layer,
								basic_cell,
								embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Unidirectional
	with tf.variable_scope("UniRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		outputs, states  = tf.nn.dynamic_rnn(cell=stack_cell,
																				dtype=tf.float32,
																				sequence_length=Seq_len,
																				inputs=X_ebd)
		cell_out = outputs[:,:-1] # Ignore the prediction of last word (period) and the prediction from last output
		cell_out_flat = tf.reshape(cell_out, [-1,hidden_size])

	# Output layer
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)), shape=[hidden_size, voc_size])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	logits = tf.matmul(cell_out_flat, w_out) + b_out
	probs = tf.nn.softmax(logits)	

	w_out_t = tf.transpose(w_out)
	return [cell_out_flat, w_out_t, b_out, probs]

def probs_BiRNN(X_id,
								Seq_len,
								Drop_keep,
								voc_size,
								embedding_size,
								hidden_size,
								num_stack_layer,
								basic_cell,
								embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell_fw]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# Output layer
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)), shape=[2*hidden_size, voc_size])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	logits = tf.matmul(cell_out_flat, w_out) + b_out
	probs = tf.nn.softmax(logits)	

	w_out_t = tf.transpose(w_out)
	return [cell_out_flat, w_out_t, b_out, probs]

def probs_UniRNN_Fc(X_id,
								Seq_len,
								Drop_keep,
								voc_size,
								embedding_size,
								hidden_size,
								num_stack_layer,
								basic_cell,
								embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Unidirectional
	with tf.variable_scope("UniRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		outputs, states  = tf.nn.dynamic_rnn(cell=stack_cell,
																				dtype=tf.float32,
																				sequence_length=Seq_len,
																				inputs=X_ebd)
		cell_out = outputs[:,:-1]
		cell_out_flat = tf.reshape(cell_out, [-1,hidden_size])

	# fully connected
	with tf.variable_scope("Fc"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[hidden_size, hidden_size])
		b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[hidden_size])
		fc = tf.nn.dropout(tf.matmul(cell_out_flat, w_) + b_, Drop_keep)

	# Output layer
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)), shape=[hidden_size, voc_size])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	logits = tf.matmul(cell_out_flat, w_out) + b_out
	probs = tf.nn.softmax(logits)	

	w_out_t = tf.transpose(w_out)
	return [cell_out_flat, w_out_t, b_out, probs]

def probs_BiRNN_Fc(X_id,
										Seq_len,
										Drop_keep,
										voc_size,
										embedding_size,
										hidden_size,
										num_stack_layer,
										basic_cell,
										embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# fully connected
	with tf.variable_scope("Fc"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, embedding_size])
		b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[embedding_size])
		fc = tf.nn.dropout(tf.matmul(cell_out_flat, w_) + b_, Drop_keep)

	# Output layer
	#w_out = tf.get_variable(name="w_out", initializer=embedding.initialized_value(), shape=[embedding_size, voc_size])
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)), shape=[embedding_size, voc_size])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	logits = tf.matmul(fc, w_out) + b_out
	probs = tf.nn.softmax(logits)	

	w_out_t = tf.transpose(w_out)
	return [fc, w_out_t, b_out, probs]


def probs_BiRNN_FcX4ReluRes(X_id,
										Seq_len,
										Drop_keep,
										voc_size,
										embedding_size,
										hidden_size,
										num_stack_layer,
										basic_cell,
										embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# fully connected
	fc = cell_out_flat
	for i in range(4):
		with tf.variable_scope("Fc"+str(i)):
			w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, 2*hidden_size])
			b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[2*hidden_size])
			fc = tf.nn.relu( fc + tf.nn.dropout(tf.matmul(fc, w_) + b_, Drop_keep) )

	with tf.variable_scope("Ebd"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, embedding_size])
		#b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[embedding_size])
		ebd = tf.matmul(fc, w_)

	# Output layer
	#w_out = tf.get_variable(name="w_out", initializer=embedding.initialized_value(), shape=[embedding_size, voc_size])
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)), shape=[embedding_size, voc_size])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	logits = tf.matmul(ebd, w_out) + b_out
	probs = tf.nn.softmax(logits)	

	w_out_t = tf.transpose(w_out)
	return [ebd, w_out_t, b_out, probs]


def probs_BiRNN_FcX4ReluResBn(X_id,
										Seq_len,
										Drop_keep,
										voc_size,
										embedding_size,
										hidden_size,
										num_stack_layer,
										basic_cell,
										embedding,
										is_train = True):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# fully connected
	fc = cell_out_flat
	for i in range(4):
		with tf.variable_scope("Fc"+str(i)) as sc:
			w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, 2*hidden_size])
			fc_branch = tf.matmul(fc, w_)
			fc_branch = tf.contrib.layers.batch_norm(fc_branch, center=True, scale=True, is_training=True, updates_collections = None, scope='bn')
			fc = tf.nn.relu(fc + fc_branch)
			sc.reuse_variables()
			tf.summary.histogram('moving_ave',tf.get_variable('bn/moving_mean',shape=[2*hidden_size]))

	with tf.variable_scope("Ebd"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, embedding_size])
		#b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[embedding_size])
		ebd = tf.matmul(fc, w_)

	# Output layer
	#w_out = tf.get_variable(name="w_out", initializer=embedding.initialized_value(), shape=[embedding_size, voc_size])
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)), shape=[embedding_size, voc_size])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	logits = tf.matmul(ebd, w_out) + b_out
	probs = tf.nn.softmax(logits)	

	w_out_t = tf.transpose(w_out)
	return [ebd, w_out_t, b_out, probs]


def probs_BiRNN_FcX4ReluDenseRes(X_id,
										Seq_len,
										Drop_keep,
										voc_size,
										embedding_size,
										hidden_size,
										num_stack_layer,
										basic_cell,
										embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# fully connected
	fc = cell_out_flat
	fc_collection = fc
	for i in range(4):
		with tf.variable_scope("Fc"+str(i)):
			w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, 2*hidden_size])
			b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[2*hidden_size])
			fc = tf.nn.relu( fc_collection + tf.nn.dropout(tf.matmul(fc, w_) + b_, Drop_keep) )
			fc_collection = fc_collection + fc

	with tf.variable_scope("Ebd"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, embedding_size])
		#b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[embedding_size])
		ebd = tf.matmul(fc, w_)

	# Output layer
	#w_out = tf.get_variable(name="w_out", initializer=embedding.initialized_value(), shape=[embedding_size, voc_size])
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)), shape=[embedding_size, voc_size])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	logits = tf.matmul(ebd, w_out) + b_out
	probs = tf.nn.softmax(logits)	

	w_out_t = tf.transpose(w_out)
	return [ebd, w_out_t, b_out, probs]


def probs_BiRNN_TieEbd(X_id,
										Seq_len,
										Drop_keep,
										voc_size,
										embedding_size,
										hidden_size,
										num_stack_layer,
										basic_cell,
										embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# fully connected
	with tf.variable_scope("Fc"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, embedding_size])
		b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[embedding_size])
		fc =tf.matmul(cell_out_flat, w_) + b_

	# Output layer
	logits = tf.matmul(fc,embedding,transpose_b=True)
	probs = tf.nn.softmax(logits)	

	return [fc, embedding, tf.zeros(shape=[voc_size]), probs]


def logits_BiRNN_Fc_Hierarchical(X_id,
										Seq_len,
										Drop_keep,
										voc_size,
										embedding_size,
										hidden_size,
										num_stack_layer,
										basic_cell,
										embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# fully connected
	with tf.variable_scope("Fc"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, 2*hidden_size])
		b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[2*hidden_size])
		fc = tf.matmul(cell_out_flat, w_) + b_

	# Output layer
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, voc_size-1])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size-1])
	logits = tf.matmul(fc, w_out) + b_out

	w_out_t = tf.transpose(w_out)
	return [fc, w_out_t, b_out, logits]

def cossimilarity_BiRNN_Fc(X_id,
											Seq_len,
											Drop_keep,
											voc_size,
											embedding_size,
											hidden_size,
											num_stack_layer,
											basic_cell,
											embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# fully connected
	with tf.variable_scope("Fc"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, 2*hidden_size])
		b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[2*hidden_size])
		fc = tf.nn.dropout(tf.matmul(cell_out_flat, w_) + b_, Drop_keep)

	# Output layer
	w_out = tf.get_variable(name="w_ebd", initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(2*hidden_size)), shape=[2*hidden_size, embedding_size])
	b_out = tf.get_variable(name="b_ebd", initializer=tf.constant_initializer(0), shape=[embedding_size])
	out_ebd = tf.matmul(fc, w_out) + b_out

	cos_similarity = tf.matmul(tf.nn.l2_normalize(out_ebd, 1, epsilon=1e-12), tf.nn.l2_normalize(embedding, 1, epsilon=1e-12), transpose_b = True)

	w_out_t = tf.transpose(w_out)
	return [out_ebd, w_out_t, b_out, cos_similarity]


def Context2Vec_BiRNN(X_id,
										Seq_len,
										Drop_keep,
										voc_size,
										embedding_size,
										hidden_size,
										num_stack_layer,
										basic_cell,
										embedding):

	# Look up word2vec embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out = tf.concat([outputs_fw[:,:-2], outputs_bw[:,2:]],2) # [batch_size, max_time_step-2, 2*hidden_num]
		cell_out_flat = tf.reshape(cell_out, [-1,2*hidden_size])

	# fully connected
	with tf.variable_scope("Fc"):
		w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, 2*hidden_size])
		b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[2*hidden_size])
		fc = tf.nn.relu(tf.matmul(cell_out_flat, w_) + b_)

	w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, embedding_size])
	b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[embedding_size])
	sentiential_ebd = tf.matmul(fc, w_) + b_

	context_embedding = tf.get_variable("context_embedding", initializer=tf.random_uniform_initializer(-1.0, 1.0),shape=[voc_size, embedding_size])
	#All_id = tf.range(0,voc_size,1,dtype=tf.int32)
	#All_word_context_ebd = tf.nn.embedding_lookup(context_embedding, All_id)
	#cosine_dis_normalized = tf.matmul(tf.nn.l2_normalize(out_ebd, 1, epsilon=1e-12), tf.nn.l2_normalize(embedding, 1, epsilon=1e-12), transpose_b = True)
	cosine_dis = tf.matmul(sentiential_ebd, context_embedding, transpose_b = True)

	return [sentiential_ebd, context_embedding, tf.constant(0.0,shape=[voc_size]), cosine_dis]

#Residual

def probs_BiRNN_MatrixFc(X_id,
										Seq_len,
										Drop_keep,
										voc_size,
										embedding_size,
										hidden_size,
										num_stack_layer,
										basic_cell,
										embedding):

	# Look up embedding
	X_ebd = tf.nn.embedding_lookup(embedding, X_id)

	# Bidirectional
	with tf.variable_scope("BiRNN"):
		attn_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=Drop_keep)
		stack_cell = tf.contrib.rnn.MultiRNNCell([attn_cell]*num_stack_layer, state_is_tuple=True)

		[outputs_fw, outputs_bw], states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_cell,
																																				cell_bw=stack_cell,
																																				dtype=tf.float32,
																																				sequence_length=Seq_len,
																																				inputs=X_ebd)
		cell_out_fw = outputs_fw[:,:-2]
		cell_out_bw = outputs_bw[:,2:]
		cell_out_fw_flat = tf.reshape(cell_out_fw, [-1,hidden_size])
		cell_out_bw_flat = tf.reshape(cell_out_bw, [-1,hidden_size])
		cell_out_flat = tf.reshape(tf.concat([cell_out_fw, cell_out_bw],2), [-1,2*hidden_size]) # [batch_size, max_time_step-2, 2*hidden_num]

	with tf.variable_scope("Structured_Fc"):
		matrixsize = 32
		fcsize = embedding_size-matrixsize
	# matrix connected
		with tf.variable_scope("Mx"):
			w_M = tf.get_variable(name="w_",
														initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(hidden_size*hidden_size)),
																																				shape=[matrixsize*hidden_size,hidden_size])
			fc_temp = tf.matmul(w_M, cell_out_fw_flat, transpose_b=True)
			fc_temp = tf.reshape(tf.transpose(fc_temp),[-1,matrixsize,hidden_size])

			fc_M = tf.matmul(fc_temp, tf.reshape(cell_out_bw_flat,[-1,hidden_size,1]) )
			fc_M = tf.reshape(fc_M,shape=[-1,matrixsize])

		# fully connected
		with tf.variable_scope("Fc"):
			w_ = tf.get_variable(name="w_", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(2*hidden_size)), shape=[2*hidden_size, fcsize])
			b_ = tf.get_variable(name="b_", initializer=tf.constant_initializer(1), shape=[fcsize])
			fc_ = tf.matmul(cell_out_flat, w_) + b_

		fc = tf.concat([fc_,fc_M],axis=1)

	# Output layer
	#w_out = tf.get_variable(name="w_out", initializer=embedding.initialized_value(), shape=[embedding_size, voc_size])
	w_out = tf.get_variable(name="w_out", initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embedding_size)), shape=[embedding_size, voc_size])
	b_out = tf.get_variable(name="b_out", initializer=tf.constant_initializer(0), shape=[voc_size])
	logits = tf.matmul(fc, w_out) + b_out
	probs = tf.nn.softmax(logits)

	w_out_t = tf.transpose(w_out)
	return [fc, w_out_t, b_out, probs]

def RNN_model(RNN_type, kwargs):
	if RNN_type == 'BiRNN':
		[ft_before_out, w_out_t, b_out, out] = probs_BiRNN(**kwargs)
	elif RNN_type == 'BiRNN_Fc':
		[ft_before_out, w_out_t, b_out, out] = probs_BiRNN_Fc(**kwargs)
	elif RNN_type == 'Ebd_BiRNN_Fc':
		[ft_before_out, w_out_t, b_out, out] = cossimilarity_BiRNN_Fc(**kwargs)
	elif RNN_type == 'BiRNN_Fc_Hierarchical':
		[ft_before_out, w_out_t, b_out, out] = logits_BiRNN_Fc_Hierarchical(**kwargs)
	elif RNN_type == 'UniRNN':
		[ft_before_out, w_out_t, b_out, out] = probs_UniRNN(**kwargs)
	elif RNN_type == 'Context2Vec_BiRNN':
		[ft_before_out, w_out_t, b_out, out] = Context2Vec_BiRNN(**kwargs)
	elif RNN_type == 'BiRNN_TieEbd':
		[ft_before_out, w_out_t, b_out, out] = probs_BiRNN_TieEbd(**kwargs)
	elif RNN_type == 'BiRNN_MatrixFc':
		[ft_before_out, w_out_t, b_out, out] = probs_BiRNN_MatrixFc(**kwargs)
	elif RNN_type == 'BiRNN_FcX4ReluRes':
		[ft_before_out, w_out_t, b_out, out] = probs_BiRNN_FcX4ReluRes(**kwargs)
	elif RNN_type == 'BiRNN_FcX4ReluDenseRes':
		[ft_before_out, w_out_t, b_out, out] = probs_BiRNN_FcX4ReluDenseRes(**kwargs)
	elif RNN_type == 'BiRNN_FcX4ReluResBn':
		[ft_before_out, w_out_t, b_out, out] = probs_BiRNN_FcX4ReluResBn(**kwargs)
	return [ft_before_out, w_out_t, b_out, out]

