import os, sys
import tensorflow as tf
import argparse
import csv
import nltk
import gensim
import numpy as np
import time

sys.path.append('utils')
import process_HTD
import guess
import model

# Data
Training_Data_Dir = 'Holmes_Training_Data'
Voc_Size = 60000 

# Model
RNN_Type = ['BiRNN', 'UniRNN','BiRNN_Fc','BiRNN_MatrixFc'][-1]
Hidden_Size = 256 
Drop_Keep = 1
Num_Stack_Layer = 2
Pre_Trained_Ebd = ['glove','fasttext'][0]

#	Training
Train_Iters = 90000
Batch_Size = 64
Learning_Rate = 0.01 
Basic_Cell = 	[tf.contrib.rnn.BasicLSTMCell(Hidden_Size, forget_bias=0.1, state_is_tuple=True),
							tf.contrib.rnn.LayerNormBasicLSTMCell(Hidden_Size, forget_bias=1.0, layer_norm=True, norm_gain=1.0, dropout_keep_prob=Drop_Keep)][0]
Loss = ['Standard','Sampled','Nce','Cosine'][1] 
Finetune_Embedding = [True,0.05]
Num_Sampled = 1024
Optimizer_Type = ['Adam','SGD'][0]
Step_Size = 39000
Sample_Rate = 0.001

# Info
Display_Step = 100
Save_Eval_Step = 2000
Eval_Batch_Size = 64
Eval_Iter = 25
Checkpoint_Dir = 'ckpt_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(RNN_Type, Pre_Trained_Ebd, Finetune_Embedding[0], Finetune_Embedding[1], Optimizer_Type, Learning_Rate, Loss, Num_Sampled, Voc_Size, Hidden_Size, Num_Stack_Layer, Drop_Keep, Sample_Rate)

def parse_args():
    parser = argparse.ArgumentParser(description='TensorFlow')
    parser.add_argument('--testpath', dest='testpath',
                        help='testing data path',
                        default="testing_data.csv", type=str)
    parser.add_argument('--outpath', dest='outpath',
                        help='output pred path',
                        default="pred.csv", type=str)
    parser.add_argument('--istrain', dest='istrain',
                        help='traing mode or testing mode',
                        default=0, type=int)

    '''if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)'''

    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = parse_args()
	print args

	# Step 1:	Load data
	HTD = process_HTD.Holmes_Training_Data()

	# Step 2:	load pretrained word2vec
	#					build word2idx, idx2word, final embedding according to intended voccabulary size, pretrained embedding, and HTD's vocabulary
	if os.path.exists('D_{}_{}.npy'.format(Pre_Trained_Ebd,Voc_Size)):
		[embedding, embedding_size, word2idx, idx2word] = np.load('D_{}_{}.npy'.format(Pre_Trained_Ebd,Voc_Size))
	else:
		embedding_model = gensim.models.Word2Vec.load_word2vec_format('pretrained_ebd/{}'.format(Pre_Trained_Ebd), binary=True)
		embedding_size = len(embedding_model['the'])
		shared_word_list = [w for w in HTD.raw_word_list if embedding_model.vocab.has_key(w)]
		assert Voc_Size <= len(shared_word_list), "Voc_Size needs to be small than word list"
		shared_word_list = shared_word_list[:Voc_Size-2]
		word2idx = {shared_word_list[i]:i+1 for i in range(len(shared_word_list))}
		word2idx.update({"<UNK>":0})
		word2idx.update({"<PAD>":len(word2idx)})
		idx2word = dict(zip(word2idx.values(), word2idx.keys()))
		embedding = np.vstack([np.random.normal(0,0.01,[embedding_size]), [embedding_model[w] for w in shared_word_list]])
		embedding = np.vstack([embedding, np.zeros(embedding_size)])
		np.save('D_{}_{}.npy'.format(Pre_Trained_Ebd,Voc_Size),[embedding, embedding_size, word2idx, idx2word])

	# Step 3:	build HTD.training_data and HTD.val_data according to pre-trained embedding
	HTD.build_data(word2idx, Sample_Rate)

	# Step 4:	construct model
	pretrained_embedding = embedding
	embedding = tf.get_variable(name="embedding",
															initializer=tf.constant_initializer(pretrained_embedding),
															shape = pretrained_embedding.shape,
															trainable = Finetune_Embedding[0])

	X_id = tf.placeholder(tf.int32,[None,None])
	Seq_len = tf.placeholder(tf.int32)
	Drop_keep = tf.placeholder(tf.float32)

	kwargs = {"X_id":X_id,
						"Seq_len":Seq_len,
						"Drop_keep":Drop_keep,
						"voc_size":Voc_Size,
						"embedding_size":embedding_size,
						"hidden_size":Hidden_Size,
						"num_stack_layer":Num_Stack_Layer,
						"basic_cell":Basic_Cell,
						"embedding":embedding}
	[ft_before_out, w_out_t, b_out, out] = model.RNN_model(RNN_Type, kwargs)

	# Step 5:	define loss, optimizer, and evalution
	Y = tf.placeholder(tf.int64)
	Mask = tf.placeholder(tf.float32)

	Y_ = Y[:,1:-1] if 'BiRNN' in RNN_Type else Y[:,1:]

	# Define loss
	if Loss == 'Nce':
		Y_flat = tf.reshape(Y_, [-1,1])
		losses = tf.nn.nce_loss(weights=w_out_t,
							         biases=b_out,
							         labels=Y_flat,
							         inputs=ft_before_out,
							         num_sampled=Num_Sampled,
							         num_classes=Voc_Size,
											 sampled_values = None)
	elif Loss == 'Sampled':
		Y_flat = tf.reshape(Y_, [-1,1])
		losses = tf.nn.sampled_softmax_loss(weights=w_out_t,
							         biases=b_out,
							         labels=Y_flat,
							         inputs=ft_before_out,
							         num_sampled=Num_Sampled,
							         num_classes=Voc_Size)
	elif Loss == 'Standard':
		Y_flat = tf.reshape(Y_, [-1])
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=Y_flat)
	elif Loss == 'Cosine':
		Y_flat = tf.reshape(Y_, [-1])
		Y_ebd = tf.nn.embedding_lookup(embedding, Y_flat)
		Y_ebd_norm = tf.nn.l2_normalize(Y_ebd, 1, epsilon=1e-12)
		out_ebd = ft_before_out
		out_ebd_norm = tf.nn.l2_normalize(out_ebd, 1, epsilon=1e-12)
		losses = 1 - tf.reduce_sum(tf.multiply(out_ebd_norm,Y_ebd_norm),1)

	Mask_ = Mask[:,1:-1] if 'BiRNN' in RNN_Type else Mask[:,1:]
	Mask_flat = tf.reshape(Mask_, [-1])

	loss_final = tf.divide(tf.reduce_sum(losses * Mask_flat), tf.cast(tf.reduce_sum(Mask_flat),tf.float32))

	# Define optimizer
	if Optimizer_Type == 'Adam':
		opt_func = tf.train.AdamOptimizer
	elif Optimizer_Type == 'SGD':
		opt_func = tf.train.GradientDescentOptimizer

	Learning_rate = tf.placeholder(tf.float32)
	global_step = tf.Variable(1, name='global_step', dtype=tf.int32, trainable=False)

	if Finetune_Embedding[0]:
		var_list1 = [tv for tv in tf.trainable_variables() if tv.name == 'embedding:0']
		assert len(var_list1) == 1, 'Wrong length of list of embedding variable, should be 1'
		var_list2 = [tv for tv in tf.trainable_variables() if tv.name != 'embedding:0']
		optr1 = opt_func(learning_rate=Learning_rate*Finetune_Embedding[1])
		optr2 = opt_func(learning_rate=Learning_rate)
		grads1 = optr1.compute_gradients(loss_final, var_list1)
		grads2 = optr1.compute_gradients(loss_final, var_list2)
		grads1 = [(tf.clip_by_value(grad, -0.5, 0.5),var) for grad, var in grads1]
		grads2 = [(tf.clip_by_value(grad, -0.5, 0.5),var) for grad, var in grads2]
		opt1 = optr1.apply_gradients(grads1,global_step=global_step)
		opt2 = optr2.apply_gradients(grads2)
		opt = tf.group(opt1, opt2)
	else:
		opt = opt_func(learning_rate = Learning_rate).minimize(loss_final,global_step=global_step)

	

	# For evaluation
	Y_eval = tf.reshape(Y_,[-1])
	Mask_eval = tf.cast(tf.not_equal(Mask_flat,0),dtype=tf.float32)
	valid_num = tf.cast(tf.reduce_sum(Seq_len),tf.float32)

	sample_idx = tf.cast(tf.reshape(tf.range(0,tf.shape(Y_eval)[0]),[-1,1]),tf.int64)
	indice = tf.concat([sample_idx,tf.reshape(Y_eval,[-1,1])],axis=1)
	probs = tf.gather_nd(out,indice)
	perplex = tf.divide(tf.reduce_sum(-tf.log(probs)*Mask_eval),valid_num)

	correct_top1 = tf.cast(tf.nn.in_top_k(out, Y_eval, 1),tf.float32)
	correct_top5 = tf.cast(tf.nn.in_top_k(out, Y_eval, 5),tf.float32)
	correct_masked_top1 = tf.multiply(correct_top1,Mask_flat)
	correct_masked_top5 = tf.multiply(correct_top5,Mask_flat)
	correct_num_top1 = tf.cast(tf.reduce_sum(correct_masked_top1),tf.float32)
	correct_num_top5 = tf.cast(tf.reduce_sum(correct_masked_top5),tf.float32)
	acc_top1 = tf.divide(correct_num_top1, valid_num)
	acc_top5 = tf.divide(correct_num_top5, valid_num)

	# Step 6:	Start process, either training mode or testing moedz
	init = tf.global_variables_initializer()

	saver = tf.train.Saver(max_to_keep=0)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=False
	with tf.Session(config=config) as sess:
		# Restore model parameter and state if exist
		os.mkdir(Checkpoint_Dir) if not os.path.exists(Checkpoint_Dir) else None
		ckpt = tf.train.get_checkpoint_state(Checkpoint_Dir)
		if ckpt and ckpt.model_checkpoint_path:
			print("restore {} ...".format(ckpt.model_checkpoint_path))
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("initiallize ...")
			sess.run(init)

		# Training mode
		if args.istrain:
			epoch_size = len(HTD.training_data) / Batch_Size
			lr = Learning_Rate
			loss_ave = 0.0
			start = time.time()
			for i in range(global_step.eval(),Train_Iters):
				inputs_padded, labels_padded, seq_len, mask_valid = HTD.get_batch(batch_size = Batch_Size)
				# Exp:
				#lr = np.exp(np.random.normal(np.log( Learning_Rate*0.1**(i/Step_Size)),1 ) if i % 100 == 0 else lr
				lr = Learning_Rate*0.1**(i/Step_Size)
				feed_dict={Learning_rate: lr, 
									X_id:inputs_padded, 
									Y:labels_padded, 
									Seq_len:seq_len, 
									Mask:mask_valid, 
									Drop_keep: Drop_Keep}
				#_, loss, step, merged_ = sess.run([opt,loss_final,global_step,merged],feed_dict=feed_dict)
				_, loss, step= sess.run([opt,loss_final,global_step],feed_dict=feed_dict)

				loss_ave += loss/Display_Step
				if step % Display_Step == 0:
					end = time.time()
					print("{}'th epoch ({}/{}), learning rate {}, loss: {}, time: {:f}".format(step/epoch_size, step, epoch_size, lr, loss_ave,end-start))
					loss_ave = 0.0
					start = time.time()

				if step % Save_Eval_Step == 0:
					print("save checkpoint_file")
					saver.save(sess,os.path.join(Checkpoint_Dir,"model"), global_step=step)

		# Testing mode, output pred.csv
		else:
			with open(args.testpath) as f:
				test_data = [l for l in csv.reader(f)]

			preds = []
			for i in range(1,len(test_data)):
				sentence = nltk.word_tokenize(test_data[i][1].lower())[:-1]
				candidate = test_data[i][2:]

				x_id = [word2idx[w] if word2idx.has_key(w) else 0 for w in sentence]
				candidata_id = [word2idx[w] if word2idx.has_key(w) else 0 for w in candidate]
				blank_pos = sentence.index('_____')

				guess_func = guess.BiRNN_guess_neighbor if 'BiRNN' in RNN_Type else guess.UniRNN_guess_neighbor
				candidata_sc = guess_func(sess,
																 out,
																 X_id,
																 Seq_len,
																 Drop_keep,
																 x_id,
																 blank_pos,
																 candidata_id,
																 window_size = 5)

				print(sentence)
				print([idx2word[cid] for cid in candidata_id])
				print(candidata_sc)
				print

				preds += ['a','b','c','d','e'][np.argmax(candidata_sc)]

			with open(args.testpath, 'wb') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(['id','answer'])
				for i in range(len(preds)):
					writer.writerow([str(i+1),preds[i]])
