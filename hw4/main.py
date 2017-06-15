import tensorflow as tf
import numpy as np
import gensim
import nltk
import os, sys
import select
from collections import OrderedDict
import time
import pprint
sys.path.append('utils')
import data_loader
import utils
sys.path.append('model')
import discriminator
import generator
import argparse

CONFIG = OrderedDict([('txtfile','data/callhome+opensub+conellsub_twitter_glove.txt'),
											('pretrain_ebd_file','glove'), # Do not change
											('voc_size',30000),
											('ebd_size',300),  # Do not change

											('cell_size',512),
											('max_time_step',25),

											('batch_size',64),
											('ebd_trainable',False),

											('lr_init',0.001),
											('lr_decay_step',20000),
											('lr_decay',0.9),
											('lr_min',0.0001)])

CONFIG.update([ ('ckpt_dir','temp/ckpt_seq2seqGAN_{}/'.format(hash(str(CONFIG)))),
								('subsample_rate',0.01),
								('ave_tfidf',True),
								('tf_rate',0.75),
								('tf_step',10000),
								('tf_local_step',3),
								('pre_train_step_G',30000),
								('pre_train_step_D',30000),
								('train_step_size',400000),
								('local_step_G',1),
								('local_step_D',2),
								('collection_batch_num', 100),
								('display_step',100),
								('save_step',1000),
								('ADDITIONAL INFO','')])

def Interaction(sess, G, D, dl):
	i, o, e = select.select( [sys.stdin], [], [], 0.01 )
	if (i) and sys.stdin.readline().strip() == 'i':
		print "Interaction mode:('exit' to back to training)"
		say = str(raw_input())
		mc = True
		rw = False
		while say != 'exit':
			if say == 'mc':
				print ">>>(using monte carlo)"
				mc = True
			elif say == 'max':
				print ">>>(using maximun likelihood)"
				mc = False
			elif say == 'r':
				rw = not rw
				print ">>>(output reward: {})".format(rw)
			else:
				x = dl.get_x(say)
				y_gen = G.generate(sess, x, monte_carlo=mc)[0]
				utter = " ".join([dl.idx2word[tok] for tok in y_gen if tok not in [0,1,2,3]])
				print ">>>botbot:",utter
				if rw:
					r = D.get_reward(sess, x, [y_gen])
					print">>>",r
			print '\n'
			say = str(raw_input())
		print "Back to training"

def Check(sess, G, D, dl, num_out=1, mc=False):
	x, y, w = dl.get_training_batch()
	r = D.get_reward(sess, x, y)
	y_gen = G.generate(sess, x, monte_carlo=mc)
	r_gen = D.get_reward(sess, x, y_gen) # * (1-tf_rate)

	pprint.pprint([" ".join([dl.idx2word[tok] for tok in xx]).replace(' <PAD>','P').replace('<EOS>','<E>').replace('<GO>','<G>') for xx in x[:num_out]])
	pprint.pprint([" ".join([dl.idx2word[tok] for tok in yy]).replace(' <PAD>','P').replace('<EOS>','<E>').replace('<GO>','<G>') for yy in y[:num_out]])
	pprint.pprint(w[:num_out])
	pprint.pprint(r[:num_out])
	pprint.pprint([" ".join([dl.idx2word[tok] for tok in yy]).replace(' <PAD>','P').replace('<EOS>','<E>').replace('<GO>','<G>') for yy in y_gen[:num_out]])
	pprint.pprint(r_gen[:num_out])

def parse_args():
	parser = argparse.ArgumentParser(description='seq2seq RL')
	parser.add_argument('--model', dest='MODEL', type=str, choices=['S2S', 'RL', 'BEST'])
	parser.add_argument('--input', dest='INPUT', type=str)
	parser.add_argument('--output', dest='OUTPUT', type=str)
	parser.add_argument('--mode', dest='MODE', type=str, default='test', choices=['train', 'test'])
	args = parser.parse_args()

	return args

if __name__ == "__main__":
	args = parse_args()

	if not os.path.exists('temp'):
		os.mkdir('temp')
	utils.make_ckpt_and_record(CONFIG['ckpt_dir'], CONFIG)

	dl = data_loader.data_loader(txtfile=CONFIG['txtfile'], subsample_rate=CONFIG['subsample_rate'], voc_size=CONFIG['voc_size'], tfidf=CONFIG['ave_tfidf'])
	D = discriminator.D(CONFIG, dl.embedding.copy())
	G = generator.G(CONFIG, dl.embedding.copy())

	if args.MODE == "train":
		saver = tf.train.Saver(max_to_keep=3)
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(CONFIG['ckpt_dir'])
			if ckpt and ckpt.model_checkpoint_path:
				print "restore {} ...".format(ckpt.model_checkpoint_path)
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				print "initiallize ..."
				sess.run(tf.global_variables_initializer())



			print("Pre-train G ...")
			t_start = time.time()
			ave_g_perplex = 0.
			step_ = G.step_mle.eval()
			while step_ < CONFIG['pre_train_step_G']:
				x, y, w = dl.get_training_batch()
				_, perplex_, step_, lr_ = G.step_MLE(sess, x, y, w)
				ave_g_perplex += perplex_/CONFIG['display_step']

				if step_ % CONFIG['display_step'] == 0:
					print "step:{} lr:{:.5f} ave_g_perplex:{:.3f} time:{:.2f}".format(step_, lr_, ave_g_perplex, time.time()-t_start)
					t_start = time.time()
					ave_g_perplex = 0.

				if step_ % CONFIG['save_step'] == 0:
					print("save checkpoint_file")
					saver.save(sess,CONFIG['ckpt_dir']+'/model_pre', global_step=step_)

				Interaction(sess, G, D, dl)
				#Check(sess, G, D, dl)

			

			print("Pre-train D ...")
			t_start = time.time()
			ave_d_loss = 0.
			step_ = D.step.eval()/2 # Beacause run twice in one iteration
			while step_ < CONFIG['pre_train_step_D']:
				x, y, w = dl.get_training_batch()
				y_gen = G.generate(sess, x, monte_carlo=np.random.binomial(1,0.5))
				_, d_loss_r, d_step_, lr_ = D.step_GAN(sess, x, y, np.ones(np.shape(y)))
				_, d_loss_f, d_step_, lr_ = D.step_GAN(sess, x, y_gen, np.zeros(np.shape(y_gen)))
				ave_d_loss += (d_loss_r + d_loss_f)/2 / CONFIG['display_step']
				step_ = d_step_/2

				if step_ % CONFIG['display_step'] == 0:
					print "step:{} lr:{:.5f} ave_d_loss:{:.3f} time:{:.2f}".format(step_, lr_, ave_d_loss, time.time()-t_start)
					t_start = time.time()
					ave_d_loss = 0.

				if step_ % CONFIG['save_step'] == 0:
					print("save checkpoint_file")
					saver.save(sess,CONFIG['ckpt_dir']+'/model_pre', global_step=step_+CONFIG['pre_train_step_G'])

				Interaction(sess, G, D, dl)
				#Check(sess, G, D, dl)

			print("train GAN ...")
			t_start = time.time()
			ave_d_loss = 0.
			ave_g_reward = 0.		
			g_step_gan_ = G.step_gan.eval()
			local_gen_data = []
			while g_step_gan_ < CONFIG['train_step_size']:
				for j in range(CONFIG['local_step_D']):
					if len(local_gen_data) == 0:
						for n in range(CONFIG["collection_batch_num"]):
							x, y, w = dl.get_training_batch()
							local_gen_data.append( (x, G.generate(sess, x, monte_carlo=True)) )

					x, y, w = dl.get_training_batch()
					_, d_loss_r, d_step_, lr_ = D.step_GAN(sess, x, y, np.ones(np.shape(y)))
					x, y_gen = local_gen_data.pop(0)
					_, d_loss_f, d_step_, lr_ = D.step_GAN(sess, x, y_gen, np.zeros(np.shape(y_gen)))
					d_loss_ = (d_loss_r + d_loss_f)/2


				tf_rate = CONFIG['tf_rate'] ** (g_step_gan_/CONFIG['tf_step'])
				teacher_forcing_step = np.random.binomial(CONFIG['tf_local_step'], tf_rate)
				for j in range(teacher_forcing_step):
					x, y, w = dl.get_training_batch()
					_, g_loss_, g_step_mle_, lr_ = G.step_MLE(sess, x, y, w)
				for j in range(CONFIG['local_step_G']):
					x, y, w = dl.get_training_batch()
					#y_ml = G.generate(sess, x, monte_carlo=False)
					#b = np.array([[sum(bb)/len(np.where(bb!=0.)[0])] for bb in D.get_reward(sess, x, y_ml)])
					y_mc = G.generate(sess, x, monte_carlo=True)
					r = D.get_reward(sess, x, y_mc)
					_, g_loss_, g_step_gan_, lr_ = G.step_GAN(sess, x, y_mc, r)

				ave_d_loss += d_loss_/CONFIG['display_step']
				ave_g_reward += np.sum(r) /len(np.where(r!=0.)) / CONFIG['display_step']


				if g_step_gan_ % CONFIG['display_step'] == 0:
					print "step:{} lr:{:.5f} ave_d_loss:{:.3f} ave_g_reward:{:.3f} tf:{:.5f}, time:{:.2f}".format(g_step_gan_, lr_, ave_d_loss, ave_g_reward, tf_rate, time.time()-t_start)
					t_start = time.time()
					ave_d_loss = 0.
					ave_g_reward = 0.

				if g_step_gan_ % CONFIG['save_step'] == 0:
					print("save checkpoint_file")
					saver.save(sess,CONFIG['ckpt_dir']+'/model_gan', global_step=g_step_gan_)
					#Check(sess, G, D, dl)

				Interaction(sess, G, D, dl)

	elif args.MODE == "test":
		with open(args.INPUT,'r') as f:
			list_of_sentence = [line.replace('\n','') for line in f.readlines()]

		saver = tf.train.Saver(max_to_keep=3)
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(CONFIG['ckpt_dir'])
			if ckpt and ckpt.model_checkpoint_path:
				if args.MODEL == "S2S" or args.MODEL == "BEST":
					print "restore {} ...".format(ckpt.all_model_checkpoint_paths[0])
					saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
				elif args.MODEL == "RL":
					print "restore {} ...".format(ckpt.all_model_checkpoint_paths[1])
					saver.restore(sess, ckpt.all_model_checkpoint_paths[1])
			else:
				print "No trained model, exit..."
				sys.exit(1)

			list_of_utterence = []
			for sent in list_of_sentence:	 
				x = dl.get_x(sent)
				y_gen = G.generate(sess, x, monte_carlo=False)[0]
				utter = " ".join([dl.idx2word[tok] for tok in y_gen if tok not in [0,1,2,3]])
				list_of_utterence.append(utter+'\n')

		with open(args.OUTPUT,'w') as f:
			f.writelines(list_of_utterence)

