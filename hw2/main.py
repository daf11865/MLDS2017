import os, sys
import tensorflow as tf
import argparse
import csv, json
import nltk
import gensim
import numpy as np
import time
import pprint

sys.path.append('utils')
import process_data
import S2VT, Attn, AttnExp2, AttnPyramid, AttnBiPyramid, BiPyramid
import bleu_eval

# Model
model = AttnPyramid.Model()

#	Training
Train_Iters = 36000
Batch_Size = 64
Learning_Rate = 0.001
Loss = ['Standard','Sampled'][0] 
Num_Sampled = 512
Optimizer_Type = ['Adam','SGD'][0]
Finetune_Embedding = 0.1
Step_Size = 24000
Sample_Rate = 1.0 # Stay tuned
Seq_Reward = 1.0
K = 2000.
Min = 1.0
Scheduled_Rate = lambda i,loss: max(K/(K+np.exp(i/K/loss)),Min)
Regularize_C_LSTM = 0

# Inference
Beam_Width = 1

# Info
Display_Step = 342
Save_Eval_Step = 342
Eval_Iter = 50
Checkpoint_Dir = 'ckpt_{}_ftebd{}_{}_lr{}_{}_ns{}_bs{}_srK{}_srM{}_sqrw{}_reg{}'.format(model.get_config_str(), Finetune_Embedding, Optimizer_Type, Learning_Rate, Loss, Num_Sampled, Batch_Size, K, Min, Seq_Reward, Regularize_C_LSTM)

def parse_args():
	parser = argparse.ArgumentParser(description='TensorFlow')
	parser.add_argument('--train', dest='train',
		                  help='train or not', default = False,
		                  type=bool)
	parser.add_argument('--testidlistpath', dest='testidlistpath',
		                  help='testing id list path',
		                  type=str)
	parser.add_argument('--featpath', dest='featpath',
		                  help='feature directory',
		                  type=str)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	print args

	# Step 1:	Load data
	print("Load data")
	VDF = process_data.Video_Data_Feat()

	# Step 2:	load pretrained word2vec, build word2idx, idx2word, pre_embedding
	print("Build word2idx, idx2word, pre_embedding")
	if os.path.exists('temp/D_{}.npy'.format('glove')):
		[pre_embedding, pre_embedding_dim, word2idx, idx2word, voc_size, ebd_size] = np.load('temp/D_{}.npy'.format('glove'))
	else:
		pre_embedding_model = gensim.models.Word2Vec.load_word2vec_format('pretrained_ebd/{}'.format(Pre_Trained_Ebd), binary=True)
		pre_embedding_dim = len(pre_embedding_model['the'])
		shared_word_list = [w[0] for w in VDF.raw_word_count if pre_embedding_model.vocab.has_key(w[0])]
		word_list = ['<PAD>','<BOS>','<EOS>','<UNK>'] + shared_word_list
		voc_size = len(word_list)
		word2idx = {word_list[i]:i for i in range(len(word_list))}
		idx2word = dict(zip(word2idx.values(), word2idx.keys()))
		pre_embedding = np.vstack([np.random.normal(0,0.01,[3,pre_embedding_dim]),[pre_embedding_model[w] for w in shared_word_list]]) # Add <BOS>,<EOS>,<UNK>
		pre_embedding = np.vstack([np.zeros(pre_embedding_dim),pre_embedding]) # Add zero embedding <PAD>
		ebd_size = pre_embedding.shape[1]
		np.save('temp/D_{}.npy'.format(Pre_Trained_Ebd),[pre_embedding, pre_embedding_dim, word2idx, idx2word, voc_size, ebd_size])

	# Step 3:	build VDF.train_data and VDF.test_data according to pre-trained embedding
	print("Build VDF data")
	VDF.build_data(word2idx, Sample_Rate)

	# Step 4:	construct model
	embedding = tf.get_variable(name="embedding",
															initializer=tf.constant_initializer(pre_embedding),
															shape = pre_embedding.shape,
															trainable = Finetune_Embedding != 0) 

	V_feat = tf.placeholder(tf.float32,[None,None,4096])
	C_idx_padded = tf.placeholder(tf.int32,[None,None])
	Sd_rate = tf.placeholder(tf.float32)

	ft_before_out, w_out_t, b_out, out = model.build_model_train(V_feat, C_idx_padded, embedding, Sd_rate)
	c_gen, c_sc = model.build_model_inference(V_feat, embedding, Beam_Width)

	# Step 5:	define loss, optimizer
	L_idx_padded = tf.placeholder(tf.int64)
	Mask_loss = tf.placeholder(tf.float32)

	# Define loss
	if Loss == 'Sampled':
		L_idx_padded_flat = tf.reshape(L_idx_padded, [-1,1])
		losses = tf.nn.sampled_softmax_loss(weights=w_out_t,
							         biases=b_out,
							         labels=L_idx_padded_flat,
							         inputs=ft_before_out,
							         num_sampled=Num_Sampled,
							         num_classes=voc_size)
	elif Loss == 'Standard':
		L_idx_padded_flat = tf.reshape(L_idx_padded, [-1])
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=L_idx_padded_flat)

	Mask_loss_flat = tf.reshape(Mask_loss, [-1])

	loss_final = tf.divide(tf.reduce_sum(losses * Mask_loss_flat), tf.cast(tf.reduce_sum(Mask_loss_flat),tf.float32))
	tf.summary.scalar('loss', loss_final)

	if Regularize_C_LSTM:
		var_list = [tv for tv in tf.trainable_variables() if "C_LSTM" in tv.name and "w" in tv.name]
		l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(Regularize_C_LSTM), weights_list=var_list)
		print("regularize: {}".format(Regularize_C_LSTM))
		for v in var_list:
			print v.name
	else:
		l2_loss = tf.zeros([1,1])

	total_loss = loss_final + l2_loss

	# Define optimizer
	if Optimizer_Type == 'Adam':
		opt_func = tf.train.AdamOptimizer
	elif Optimizer_Type == 'SGD':
		opt_func = tf.train.GradientDescentOptimizer

	Learning_rate = tf.placeholder(tf.float32)
	global_step = tf.Variable(1, name='global_step', dtype=tf.int32, trainable=False)

	if Finetune_Embedding != 0:
		var_list1 = [tv for tv in tf.trainable_variables() if tv.name == 'embedding:0']
		assert len(var_list1) == 1, 'Wrong length of list of embedding variable, should be 1'
		var_list2 = [tv for tv in tf.trainable_variables() if tv.name != 'embedding:0']
		optr1 = opt_func(learning_rate=Learning_rate*Finetune_Embedding)
		optr2 = opt_func(learning_rate=Learning_rate)
		grads1 = optr1.compute_gradients(total_loss, var_list1)
		grads2 = optr1.compute_gradients(total_loss, var_list2)
		#[tf.summary.histogram(var.name, grad) for grad, var in grads1]
		#[tf.summary.histogram(var.name, grad) for grad, var in grads2]
		#grads1 = [(tf.clip_by_value(grad, -0.5, 0.5),var) for grad, var in grads1]
		#grads2 = [(tf.clip_by_value(grad, -0.5, 0.5),var) for grad, var in grads2]
		opt1 = optr1.apply_gradients(grads1,global_step=global_step)
		opt2 = optr2.apply_gradients(grads2)
		opt = tf.group(opt1, opt2)
	else:
		opt = opt_func(learning_rate = Learning_rate).minimize(total_loss,global_step=global_step)

	# Step 6:	Start process, either training mode or testing moedz
	init = tf.global_variables_initializer()

	saver = tf.train.Saver(max_to_keep=3)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=False
	with tf.Session(config=config) as sess:
		# TensorBoard
		#merged = tf.summary.merge_all()
		#train_writer = tf.summary.FileWriter(Checkpoint_Dir+'/log', sess.graph)

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
		if args.train:
			epoch_size = len(VDF.train_data_format2)/ Batch_Size
			lr = Learning_Rate
			loss_ave = 0.0
			loss_ave_temp = 10
			start = time.time()
			for step in range(global_step.eval(),Train_Iters):
				v_feat, c_idx_padded, l_idx_padded, c_idx_len, loss_mask = VDF.get_train_batch(batch_size=Batch_Size, seq_reward=Seq_Reward)
				c_idx_padded, l_idx_padded, loss_mask = model.train_data_second_process(c_idx_padded, l_idx_padded, loss_mask)
				lr = Learning_Rate*0.1**(step/Step_Size)
				feed_dict={Learning_rate: lr, 
									V_feat:v_feat,
									C_idx_padded:c_idx_padded, 
									L_idx_padded:l_idx_padded, 
									Mask_loss:loss_mask,
									Sd_rate:Scheduled_Rate(step, loss_ave_temp)}

				#_, loss, merged_ = sess.run([opt,loss_final,merged],feed_dict=feed_dict)
				_, loss, l2_loss_= sess.run([opt,loss_final,l2_loss],feed_dict=feed_dict)

				if step % Save_Eval_Step == 0:
					if step % (5*Save_Eval_Step):
						print("save checkpoint_file")
						saver.save(sess,os.path.join(Checkpoint_Dir,"model"), global_step=step)

					ave_score = [0.]*Beam_Width
					max_ave_score = [0.]*Beam_Width
					for j in range(Eval_Iter):
						v_feat, c_refs, v_idx = VDF.get_test_batch(batch_size = 1, nth = j)
						c_gen_, c_sc_ = sess.run([c_gen, c_sc],feed_dict={V_feat: v_feat}) # [time,batch,word]

						c_gen_ = np.array(c_gen_)
						c_refs = c_refs[0]

						for k in range(len(c_gen_[:1])):
							caption_generated = []
							for idx in c_gen_[k]:
								if idx == 2:
									break
								caption_generated.append(idx2word[idx])
							caption_generated = " ".join(caption_generated)
							print v_idx[0], caption_generated
							score, max_score = bleu_eval.BLEU(caption_generated, c_refs)
							ave_score[k] += score/Eval_Iter
							max_ave_score[k] += max_score/Eval_Iter

					print("BLEU ave: {} max: {}".format(["%.3f"%sc for sc in ave_score], ["%.3f"%sc for sc in max_ave_score]))
					with open(os.path.join(Checkpoint_Dir,"BLEU.txt"),'a') as fw:
						fw.write("Step: {}, Loss: {:.3f} BLEU ave: {} max: {}\n".format(step,loss_ave,["%.3f"%sc for sc in ave_score], ["%.3f"%sc for sc in max_ave_score]))

				loss_ave += loss/Display_Step
				if step % Display_Step == 0:
					end = time.time()
					print("{}'th epoch ({}/{}), learning rate {}, loss: {}({}), scgr: {}, time: {:f}".format(step/epoch_size, step, epoch_size, lr, loss_ave, l2_loss_, Scheduled_Rate(step,loss_ave_temp), end-start))
					loss_ave_temp = loss_ave
					loss_ave = 0.0
					start = time.time()
					#train_writer.add_summary(merged_, step)

		# Inference mode
		if args.testidlistpath is None:
			ave_score = [0.]*Beam_Width
			max_ave_score = [0.]*Beam_Width
			for j in range(Eval_Iter):
				v_feat, c_refs, v_idx = VDF.get_test_batch(batch_size = 1, nth = j)
				c_gen_, c_sc_ = sess.run([c_gen, c_sc],feed_dict={V_feat: v_feat})

				c_gen_ = np.array(c_gen_)
				c_refs = c_refs[0]

				print v_idx[0],":"
				for k in range(len(c_gen_)):
					caption_generated = []
					for idx in c_gen_[k]:
						if idx == 2:
							break
						caption_generated.append(idx2word[idx])
					caption_generated = " ".join(caption_generated)
					print caption_generated
					score, max_score = bleu_eval.BLEU(caption_generated, c_refs)
					ave_score[k] += score/Eval_Iter
					max_ave_score[k] += max_score/Eval_Iter
			print("BLEU ave: {} max: {}".format(ave_score[:3], max_ave_score[:3]))

		if args.testidlistpath and args.featpath:
			res = {}
			for feat_npy in os.listdir(args.featpath):
				idd = feat_npy.replace('.npy','')
				feat = np.load(os.path.join(args.featpath,feat_npy))
				c_gen_, c_sc_ = sess.run([c_gen, c_sc],feed_dict={V_feat: [feat]})

				caption = " ".join([idx2word[idx] for idx in c_gen_[0] if idx != 2])
				res[idd] = caption

			with open(args.testidlistpath) as f:
				id_order = f.readlines()
				id_order = [l.replace('\n',"") for l in id_order]
			res_json = [{"caption":res[iid],"id":iid} for iid in id_order]
			
			with open('output.json','w') as fw:
				json.dump(res_json, fw, indent=2)
