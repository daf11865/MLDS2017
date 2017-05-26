import os, sys
import scipy.misc
import numpy as np
import pprint
from collections import OrderedDict
import tensorflow as tf
sys.path.append('utils')
import ops
import face_data
import utils
import time
import tensorflow.contrib.layers as tfly
import gc
import random
gc.disable()

ACTIVATION = {None:None, 'lrelu':ops.lrelu, 'elu':tf.nn.elu}
NORMALIZER = {None:None, 'bn':tfly.batch_norm}
OUT_ACTIVATION = {None:None, 'lrelu':ops.lrelu, 'elu':tf.nn.elu}
OUT_NORMALIZER = {None:None, 'bn':tfly.batch_norm}

CONFIG = OrderedDict([('use_tagless_data',False),
											('sample_tagless_data',False),
											('sub_mean',True),
											('image_size_I',64),
											('image_size_II',96),
											('flip',True),
											('z_dim',128),
											('z_normal',True),
											('t_dim',64),
											('t_aug',True),
											('t_aug_penalty',True),

											('h_voc_size',13), # Do not change
											('e_voc_size',12), # Do not change


											('g_conv_depth',512),
											('d_conv_depth',512),
											('INOR','1.2 * d_loss_r, D +1 conv, add judge D'),

											('d_loc_step_I',1), # Do not change
											('g_loc_step_I',1),
											('d_loc_step_II',1), # Do not change
											('g_loc_step_II',1),

											('AdamOpt',True),

											('lr_init',0.0001),
											('lr_decay',1.0),
											('lr_decay_step',20000),
											('lr_min',0.00005),
											('batch_size',32)])

CONFIG.update([ ('ckpt_dir','temp/ckpt_ConStackLSGANMD_{}/'.format(hash(str(CONFIG)))),
								('train_step_I',51000),
								('train_step_II',99000),
								('display_step',100),
								('save_step',1000),
								('ADDITIONAL INFO','')])

STD = 0.350

def make_ckpt_and_record(dir, info):
	if not os.path.exists(dir):
		os.mkdir(dir)
	with open(os.path.join(dir,'config.txt'), 'w') as f:
		for key, value in info.items():
			f.write('%s:%s\n' % (key, value))

def save_image(ims, step, gen_dir):
	os.mkdir(gen_dir) if not os.path.exists(gen_dir) else None
	side_num = int(np.ceil(np.sqrt(ims.shape[0])))
	h, w = ims.shape[1], ims.shape[2]
	tiled_img = np.zeros((h * side_num, w * side_num, 3))
	for idx in range(len(ims)):
		i = idx % side_num
		j = idx // side_num
		tiled_img[j*h:j*h+h, i*w:i*w+w, :] = ims[idx]
	scipy.misc.imsave(os.path.join(gen_dir, '{}.jpg'.format(step)),tiled_img)

class gan:
	def __init__(self):
		print('Conditional Stack Least Square DCGAN')

		t_h = tf.placeholder('int32', [CONFIG['batch_size']])
		t_e = tf.placeholder('int32', [CONFIG['batch_size']])
		z = tf.placeholder('float32', [CONFIG['batch_size'], CONFIG['z_dim']])
		x_real64x64 = tf.placeholder('float32', [CONFIG['batch_size'], 64, 64, 3])
		x_wrong64x64 = tf.placeholder('float32', [CONFIG['batch_size'], 64, 64, 3])
		x_real96x96 = tf.placeholder('float32', [CONFIG['batch_size'], 96, 96, 3])
		x_wrong96x96 = tf.placeholder('float32', [CONFIG['batch_size'], 96, 96, 3])
		is_train = tf.placeholder('bool')

		self.placeholder = {'t_h':t_h,
												't_e':t_e,
												'z':z,
												'x_real64x64':x_real64x64,
												'x_wrong64x64':x_wrong64x64,
												'x_real96x96':x_real96x96,
												'x_wrong96x96':x_wrong96x96,
												'is_train':is_train}
		self.data = face_data.data(CONFIG)

	def condition_aug_param(self, t_ebd, out_dim, scope):
		with tf.variable_scope(scope): 
			mean = ops.linear(t_ebd, out_dim, 't_aug_mean', w_initializer=tf.random_normal_initializer(0, 0.02))
			log_std = ops.linear(t_ebd, out_dim, 't_aug_std', w_initializer=tf.random_normal_initializer(0, 0.02))
		return mean,  log_std

	def condition(self, h_idx, e_idx, scope = 'condition'):
		with tf.variable_scope(scope): 
			t_h_ebd_param = tf.get_variable(name="t_h_ebd",
																initializer=tf.random_normal_initializer(0, 1),
																shape = [CONFIG['h_voc_size'], CONFIG['t_dim']/2])
			t_e_ebd_param = tf.get_variable(name="t_e_ebd",
																initializer=tf.random_normal_initializer(0, 1),
																shape = [CONFIG['e_voc_size'], CONFIG['t_dim']/2])

			t_h_ebd = tf.nn.embedding_lookup(t_h_ebd_param, h_idx)
			t_e_ebd = tf.nn.embedding_lookup(t_e_ebd_param, e_idx)
		return tf.concat([t_h_ebd,t_e_ebd], 1)






	################ Stage I Model ################
	def generatorI(self, z, t_g, is_train, scope = 'generatorI'):
		with tf.variable_scope(scope): 
			size = CONFIG['image_size_I']/16
			depth = CONFIG['g_conv_depth']
			#w_initializer=tf.random_normal_initializer(0, 0.02)

			zt = tf.concat([z, t_g], axis = 1)
			maps = tf.reshape(ops.linear(zt, depth*size*size, 'ebd2map', activation = tf.nn.relu), [-1,size,size,depth])
			maps = ops.deconv(maps, depth/2, 'map1', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=tf.nn.relu)
			maps = ops.deconv(maps, depth/4, 'map2', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=tf.nn.relu)
			maps = ops.deconv(maps, depth/8, 'map3', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=tf.nn.relu)
			g_out = ops.deconv(maps, 3, 'map4', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=tf.tanh)
			g_out = g_out * 0.5 + 0.5 - self.data.im_mean
		return g_out

	def discriminatorI(self, x, t_ebd, is_train, reuse = False, scope = 'discriminatorI'):
		with tf.variable_scope(scope, reuse=reuse): 
			depth = CONFIG['d_conv_depth']

			maps = ops.conv(x, depth/8, 'map1', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=ops.lrelu)
			maps = ops.conv(maps, depth/4, 'map2', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=ops.lrelu)
			maps = ops.conv(maps, depth/2, 'map3', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=ops.lrelu)
			maps = ops.conv(maps, depth/1, 'map4', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=ops.lrelu)
			maps = ops.conv(maps, depth/1, 'map5', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=ops.lrelu)

			size = maps.get_shape().as_list()[1]
			t_ebd = tf.expand_dims(t_ebd,1)
			t_ebd = tf.expand_dims(t_ebd,2)
			tiled = tf.tile(t_ebd, [1,size,size,1])
			maps = tf.concat([maps, tiled],axis = -1)

			maps = ops.conv(maps, depth/1, 'mapt5', k=1, s=1, normalizer=tfly.batch_norm, activation=ops.lrelu)
			out = ops.linear(tf.reshape(maps,[CONFIG['batch_size'],-1]), 1, 'logits', activation=None)#tf.nn.sigmoid)
		return out
		



	################ Stage II Model ################
	def generatorII(self, x, t_aug, is_train, scope = 'generatorII'):
		with tf.variable_scope(scope): 
			depth = CONFIG['g_conv_depth']
			residual = True

			with tf.variable_scope('DownSampling'): 
				maps = ops.conv(x, depth/2, 'map1', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=tf.nn.relu)
				maps = ops.conv(maps, depth/2, 'map1-0', k=3, s=1, normalizer=tfly.batch_norm, is_train=is_train, residual=residual, activation=tf.nn.relu)
				maps = ops.conv(maps, depth/1, 'map2', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=tf.nn.relu)
				maps = ops.conv(maps, depth/1, 'map2-0', k=3, s=1, normalizer=tfly.batch_norm, is_train=is_train, residual=residual, activation=tf.nn.relu)

			with tf.variable_scope('MiddleLayer'): 
				size = maps.get_shape().as_list()[1]
				t_aug = tf.expand_dims(t_aug,1)
				t_aug = tf.expand_dims(t_aug,2)
				tiled = tf.tile(t_aug, [1,size,size,1])
				maps = tf.concat([maps, tiled],axis = -1)

				maps = ops.conv(maps, depth/1, 'mapt2', k=3, s=1, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=tf.nn.relu)
				maps = ops.conv(maps, depth/1, 'mapt2_0', k=3, s=1, normalizer=tfly.batch_norm, is_train=is_train, residual=residual, activation=tf.nn.relu)

			with tf.variable_scope('UpSampling'): 
				maps = ops.deconv(maps, depth/2, 'map1', k=5, s=3, normalizer=tfly.batch_norm, is_train=is_train, activation=tf.nn.relu)
				maps = ops.deconv(maps, depth/4, 'map2', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, activation=tf.nn.relu)
				g_out = ops.deconv(maps, 3, 'map2_0', k=3, s=1, normalizer=tfly.batch_norm, is_train=is_train, activation=tf.tanh)
			g_out = g_out * 0.5 + 0.5 - self.data.im_mean
		return g_out

	def discriminatorII(self, x, t_ebd, is_train, reuse = False, scope = 'discriminatorII'):
		with tf.variable_scope(scope, reuse=reuse): 
			depth = CONFIG['d_conv_depth']
			residual = True

			maps = ops.conv(x, depth/16, 'map1', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=ops.lrelu)
			maps = ops.conv(maps, depth/8, 'map2', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=ops.lrelu)
			maps = ops.conv(maps, depth/4, 'map3', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=ops.lrelu)
			maps = ops.conv(maps, depth/2, 'map4', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=ops.lrelu)
			maps = ops.conv(maps, depth/1, 'map5', k=3, s=2, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=ops.lrelu)

			size = maps.get_shape().as_list()[1]
			t_ebd = tf.expand_dims(t_ebd,1)
			t_ebd = tf.expand_dims(t_ebd,2)
			tiled = tf.tile(t_ebd, [1,size,size,1])
			maps = tf.concat([maps, tiled], axis = -1)

			maps = ops.conv(maps, depth/1, 'map5_0', k=1, s=1, normalizer=tfly.batch_norm, is_train=is_train, residual=False, activation=ops.lrelu)
			out = ops.linear(tf.reshape(maps,[CONFIG['batch_size'],-1]), 1, 'logits', activation=None)
		return out







	def build_stageI(self, z, t_ebd, x_real, x_wrong, is_train):
		with tf.variable_scope('StageI'): 
			if CONFIG['t_aug']:
				mean, log_std = self.condition_aug_param(t_ebd, CONFIG['t_dim'], 'generatorI')
				epsilon = tf.truncated_normal(tf.shape(mean),0.0,STD)
				t_gen = mean + epsilon * tf.exp(log_std)
			else:
				t_gen = ops.linear(t_ebd, CONFIG['t_dim'], 'generatorI')
			g_out = self.generatorI(z, t_gen, is_train, scope='generatorI')

			d_out_fake = self.discriminatorI(g_out, t_ebd, is_train, scope='discriminatorI')
			d_out_real = self.discriminatorI(x_real, t_ebd, is_train, reuse = True, scope='discriminatorI')
			d_out_wrong = self.discriminatorI(x_wrong, t_ebd, is_train, reuse = True, scope='discriminatorI')

			# Define loss
			if CONFIG['t_aug']:
				g_loss_c = tf.reduce_mean(-log_std + 0.5*(-1 + tf.exp(2. * log_std) + tf.square(mean)))
			g_loss_g = tf.reduce_mean(tf.abs(d_out_fake-1))
			g_loss = g_loss_g + g_loss_c if CONFIG['t_aug'] else g_loss_g

			d_loss_r = tf.reduce_mean(tf.square(d_out_real-1)/2)
			d_loss_f = tf.reduce_mean(tf.square(d_out_fake)/2)
			d_loss_w = tf.reduce_mean(tf.square(d_out_wrong)/2)
			d_loss = 1.2 * d_loss_r + d_loss_f + d_loss_w

			# Define optmizer
			opt_func = tf.train.AdamOptimizer if CONFIG['AdamOpt'] else tf.train.RMSPropOptimizer
			global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)
			lr = tf.maximum(tf.train.exponential_decay(CONFIG['lr_init'], global_step, CONFIG['lr_decay_step'], CONFIG['lr_decay'], staircase=True),CONFIG['lr_min'])

			g_var = [v for v in tf.trainable_variables() if 'generatorI' in v.name]# or 'condition' in v.name]
			d_var = [v for v in tf.trainable_variables() if 'discriminatorI' in v.name]# or 'condition' in v.name]
			pprint.pprint([(v.name, v.get_shape().as_list()) for v in g_var])
			pprint.pprint([(v.name, v.get_shape().as_list()) for v in d_var])
			g_dep = [up for up in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'generatorI' in up.name]
			d_dep = [up for up in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminatorI' in up.name]
			with tf.control_dependencies(d_dep):
				d_opt = opt_func(lr).minimize(d_loss, var_list=d_var, global_step = global_step)
			with tf.control_dependencies(g_dep):
				g_opt = opt_func(lr).minimize(g_loss, var_list=g_var)

			# pre train condition
			d_pre_out_real = self.discriminatorI(x_real, t_ebd, is_train, scope='discriminatorI_pre')
			d_pre_out_wrong = self.discriminatorI(x_wrong, t_ebd, is_train, reuse = True, scope='discriminatorI_pre')
			d_pre_loss_r = tf.reduce_mean(tf.square(d_pre_out_real-1)/2)
			d_pre_loss_w = tf.reduce_mean(tf.square(d_pre_out_wrong)/2)
			d_pre_loss = d_pre_loss_r + d_pre_loss_w
			d_pre_var = [v for v in tf.trainable_variables() if 'discriminatorI_pre' in v.name or 'condition' in v.name]
			pprint.pprint([(v.name, v.get_shape().as_list()) for v in d_pre_var])
			d_pre_opt = opt_func(lr).minimize(d_pre_loss, var_list=d_pre_var)	

			# judge D
			judge_out_fake = self.discriminatorI(g_out, t_ebd, is_train, scope='discriminatorI_judge')
			judge_out_real = self.discriminatorI(x_real, t_ebd, is_train, reuse = True, scope='discriminatorI_judge')
			judge_out_wrong = self.discriminatorI(x_wrong, t_ebd, is_train, reuse = True, scope='discriminatorI_judge')
			judge_out = judge_out_real
			judge_loss_r = tf.reduce_mean(tf.square(judge_out_real-1)/2)
			judge_loss_f = tf.reduce_mean(tf.square(judge_out_fake)/2)
			judge_loss_w = tf.reduce_mean(tf.square(judge_out_wrong)/2)
			judge_loss = 1.2 * judge_loss_r + judge_loss_f + judge_loss_w
			judge_var = [v for v in tf.trainable_variables() if 'discriminatorI_judge' in v.name]
			pprint.pprint([(v.name, v.get_shape().as_list()) for v in judge_var])
			judge_dep = [up for up in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminatorI_judge' in up.name]
			with tf.control_dependencies(judge_dep):
				judge_opt = opt_func(lr).minimize(judge_loss, var_list=judge_var)

			re = {'g_out':g_out, 
						'g_loss':g_loss, 
						'd_loss':d_loss, 
						'd_loss_r':d_loss_r,
						'd_loss_f':d_loss_f,
						'd_loss_w':d_loss_w,
						'global_step':global_step, 
						'lr':lr,
						'd_opt':d_opt, 
						'g_opt':g_opt,
						'd_pre_opt':d_pre_opt,
						'd_pre_loss':d_pre_loss,
						'judge':-judge_out,
						'judge_opt':judge_opt}
			return re

	def build_stageII(self, x, t_ebd, x_real, x_wrong, is_train):
		with tf.variable_scope('StageII'): 
			if CONFIG['t_aug']:
				mean, log_std = self.condition_aug_param(t_ebd, CONFIG['t_dim'], 'generatorII')
				epsilon = tf.truncated_normal(tf.shape(mean),0.0,STD)
				t_gen = mean + epsilon * tf.exp(log_std)
			else:
				t_gen = ops.linear(t_ebd, CONFIG['t_dim'], 'generatorII')

			g_out = self.generatorII(x, t_gen, is_train, scope='generatorII')

			d_out_fake = self.discriminatorII(g_out, t_ebd, is_train, scope='discriminatorII')
			d_out_real = self.discriminatorII(x_real, t_ebd, is_train, reuse = True, scope='discriminatorII')
			d_out_wrong = self.discriminatorII(x_wrong, t_ebd, is_train, reuse = True, scope='discriminatorII')

			# Define loss
			if CONFIG['t_aug']:
				g_loss_c = tf.reduce_mean(-log_std + 0.5*(-1 + tf.exp(2. * log_std) + tf.square(mean)))
			g_loss_g = tf.reduce_mean(tf.abs(d_out_fake-1))
			g_loss = g_loss_g + g_loss_c if CONFIG['t_aug'] else g_loss_g

			d_loss_r = tf.reduce_mean(tf.square(d_out_real-1)/2)
			d_loss_f = tf.reduce_mean(tf.square(d_out_fake)/2)
			d_loss_w = tf.reduce_mean(tf.square(d_out_wrong)/2)
			d_loss = 1.2 * d_loss_r + d_loss_f + d_loss_w

			# Define optmizer
			opt_func = tf.train.AdamOptimizer if CONFIG['AdamOpt'] else tf.train.RMSPropOptimizer
			global_step = tf.Variable(0, name='global_step', dtype=tf.int32, trainable=False)
			lr = tf.maximum(tf.train.exponential_decay(CONFIG['lr_init'], global_step, CONFIG['lr_decay_step'], CONFIG['lr_decay'], staircase=True),CONFIG['lr_min'])

			g_var = [v for v in tf.trainable_variables() if 'generatorII' in v.name]
			d_var = [v for v in tf.trainable_variables() if 'discriminatorII' in v.name]
			pprint.pprint([(v.name, v.get_shape().as_list()) for v in g_var])
			pprint.pprint([(v.name, v.get_shape().as_list()) for v in d_var])
			g_dep = [up for up in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'generatorII' in up.name]
			d_dep = [up for up in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminatorII' in up.name]
			with tf.control_dependencies(d_dep):
				d_opt = opt_func(lr).minimize(d_loss, var_list=d_var, global_step = global_step)
			with tf.control_dependencies(g_dep):
				g_opt = opt_func(lr).minimize(g_loss, var_list=g_var)

			# judge D
			judge_out_fake = self.discriminatorII(g_out, t_ebd, is_train, scope='discriminatorII_judge')
			judge_out_real = self.discriminatorII(x_real, t_ebd, is_train, reuse = True, scope='discriminatorII_judge')
			judge_out_wrong = self.discriminatorII(x_wrong, t_ebd, is_train, reuse = True, scope='discriminatorII_judge')
			judge_out = judge_out_real
			judge_loss_r = tf.reduce_mean(tf.square(judge_out_real-1)/2)
			judge_loss_f = tf.reduce_mean(tf.square(judge_out_fake)/2)
			judge_loss_w = tf.reduce_mean(tf.square(judge_out_wrong)/2)
			judge_loss = 1.2 * judge_loss_r + judge_loss_f + judge_loss_w
			judge_var = [v for v in tf.trainable_variables() if 'discriminatorII_judge' in v.name]
			pprint.pprint([(v.name, v.get_shape().as_list()) for v in judge_var])
			judge_dep = [up for up in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminatorII_judge' in up.name]
			with tf.control_dependencies(judge_dep):
				judge_opt = opt_func(lr).minimize(judge_loss, var_list=judge_var)

			re = {'g_out':g_out, 
						'g_loss':g_loss, 
						'd_loss':d_loss, 
						'global_step':global_step, 
						'lr':lr,
						'd_opt':d_opt, 
						'g_opt':g_opt,
						'judge':-judge_out,
						'judge_opt':judge_opt}
			return re





	def build_model(self):
		# Define model
		t_ebd = self.condition(self.placeholder['t_h'], self.placeholder['t_e'])
		self.tebd = t_ebd
		self.stageI = self.build_stageI(self.placeholder['z'], t_ebd, self.placeholder['x_real64x64'], self.placeholder['x_wrong64x64'], self.placeholder['is_train'])
		self.stageII = self.build_stageII(self.stageI['g_out'], t_ebd, self.placeholder['x_real96x96'], self.placeholder['x_wrong96x96'], self.placeholder['is_train'])



	def train(self):
		utils.make_ckpt_and_record(CONFIG['ckpt_dir'], CONFIG)

		with tf.Session() as sess:
			saver = tf.train.Saver(max_to_keep=3)
			ckpt = tf.train.get_checkpoint_state(CONFIG['ckpt_dir'])
			if ckpt and ckpt.model_checkpoint_path:
				print "restore {} ...".format(ckpt.model_checkpoint_path)
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				print "initiallize ..."
				sess.run(tf.global_variables_initializer())

				# Stage I condition pre-training
				print 'Stage I condition pre-training...'
				ave_d_pre_loss = 0.
				for i in range(10000):
					batch = self.data.get_training_batch() 
					fetch = [self.stageI['d_pre_opt'],
									self.stageI['d_pre_loss']]
					_, d_pre_loss_ = sess.run(fetch, feed_dict={self.placeholder['t_h']:batch['t_h'],
																											self.placeholder['t_e']:batch['t_e'],
																											self.placeholder['x_real64x64']:batch['x_real64x64'],
																											self.placeholder['x_wrong64x64']:batch['x_wrong64x64'],
																											self.placeholder['is_train']:True})
					ave_d_pre_loss += d_pre_loss_ / CONFIG['save_step']

					if i % CONFIG['save_step'] == 0:
						print('{}: condition loss:{:.5f}'.format(i, ave_d_pre_loss))
						ave_d_pre_loss = 0.

				print("save checkpoint_file")
				saver.save(sess,CONFIG['ckpt_dir']+'/model', global_step=0)

			# Stage I training
			print 'Stage I ...'
			ave_d_loss = 0.0
			ave_g_loss = 0.0
			ave_d_loss_r = 0.0
			ave_d_loss_f = 0.0
			ave_d_loss_w = 0.0
			global_step_ = self.stageI['global_step'].eval()
			start = time.time()
			while global_step_ <= CONFIG['train_step_I']:
				for loc_step in range(CONFIG['d_loc_step_I']) if global_step_>0 else range(100):
					batch = self.data.get_training_batch() 
					fetch = [self.stageI['d_opt'],
									self.stageI['d_loss'],
									self.stageI['d_loss_r'],
									self.stageI['d_loss_f'],
									self.stageI['d_loss_w'],
									self.stageI['global_step'],
									self.stageI['lr'],
									self.stageI['judge_opt']]
					_, d_loss_, d_loss_r_, d_loss_f_, d_loss_w_, global_step_, lr_, _ = sess.run(fetch, feed_dict={self.placeholder['z']:batch['z'],
																																																					self.placeholder['t_h']:batch['t_h'],
																																																					self.placeholder['t_e']:batch['t_e'],
																																																					self.placeholder['x_real64x64']:batch['x_real64x64'],
																																																					self.placeholder['x_wrong64x64']:batch['x_wrong64x64'],
																																																					self.placeholder['is_train']:True})
				ave_d_loss += d_loss_/CONFIG['display_step']
				ave_d_loss_r += d_loss_r_/CONFIG['display_step']
				ave_d_loss_f += d_loss_f_/CONFIG['display_step']
				ave_d_loss_w += d_loss_w_/CONFIG['display_step']

				for loc_step in range(CONFIG['g_loc_step_I']):
					batch = self.data.get_training_batch() 
					fetch = [self.stageI['g_opt'],
									self.stageI['g_loss']]
					_, g_loss_ = sess.run(fetch, feed_dict={self.placeholder['z']:batch['z'],
																									self.placeholder['t_h']:batch['t_h'],
																									self.placeholder['t_e']:batch['t_e'],
																									self.placeholder['is_train']:True})
				ave_g_loss += g_loss_/CONFIG['display_step']

				if global_step_ % CONFIG['display_step'] == 0:
					end = time.time()
					print '{}: lr:{:.7f} D loss:{:.5f}({:.5f}, {:.5f}, {:.5f}), G loss:{:.5f}, time:{:.3f}'.format(global_step_, lr_ ,ave_d_loss, ave_d_loss_r, ave_d_loss_f, ave_d_loss_w, ave_g_loss, end-start)
					start = time.time()
					ave_d_loss = 0.0
					ave_g_loss = 0.0
					ave_d_loss_r = 0.0
					ave_d_loss_f = 0.0
					ave_d_loss_w = 0.0

				if global_step_ % CONFIG['save_step'] == 0:
					gen_testmd = []
					for k in range(8):
						val_t_h, val_t_e, val_noise = self.data.get_val_data2(k)
						[g_out_] = sess.run([self.stageII['g_out']], feed_dict={self.placeholder['z']:val_noise,
																																	self.placeholder['t_h']:val_t_h,
																																	self.placeholder['t_e']:val_t_e,
																																	self.placeholder['is_train']:False})
						[judge] = sess.run([self.stageII['judge']], feed_dict={self.placeholder['t_h']:val_t_h,
																																	self.placeholder['t_e']:val_t_e,
																																	self.placeholder['x_real96x96']:g_out_,
																																	self.placeholder['is_train']:False})
						min_idx = np.reshape(judge,[-1]).argsort()[:8]
						gen_testmd.append(g_out_[min_idx].copy()+self.data.im_mean)
					gen_testmd = np.reshape(gen_testmd, [-1,64,64,3])
					utils.save_image(gen_testmd, global_step_, os.path.join(CONFIG['ckpt_dir'],'gen_I_bntest'))

					print("save checkpoint_file")
					saver.save(sess,CONFIG['ckpt_dir']+'/model', global_step=global_step_)

			# Stage II training
			print 'Stage II ...'
			ave_d_loss = 0.0
			ave_g_loss = 0.0
			global_step_ = self.stageII['global_step'].eval()
			start = time.time()
			while global_step_ <= CONFIG['train_step_II']:
				for loc_step in range(CONFIG['d_loc_step_II']) if global_step_>0 else range(100):
					batch = self.data.get_training_batch() 
					fetch = [self.stageII['d_opt'],
									self.stageII['d_loss'],
									self.stageII['global_step'],
									self.stageII['lr'],
									self.stageII['judge_opt']]
					_, d_loss_, global_step_, lr_, _ = sess.run(fetch, feed_dict={self.placeholder['z']:batch['z'],
																																	 	self.placeholder['t_h']:batch['t_h'],
																																		self.placeholder['t_e']:batch['t_e'],
																																		self.placeholder['x_real96x96']:batch['x_real96x96'],
																																		self.placeholder['x_wrong96x96']:batch['x_wrong96x96'],
																																		self.placeholder['is_train']:True})

				for loc_step in range(CONFIG['g_loc_step_II']):
					batch = self.data.get_training_batch()
					fetch = [self.stageII['g_opt'],
									self.stageII['g_loss']]
					_, g_loss_ = sess.run(fetch, feed_dict={self.placeholder['z']:batch['z'],
																								 	self.placeholder['t_h']:batch['t_h'],
																									self.placeholder['t_e']:batch['t_e'],
																									self.placeholder['is_train']:True})

				ave_d_loss += d_loss_/CONFIG['display_step']
				ave_g_loss += g_loss_/CONFIG['display_step']
				if global_step_ % CONFIG['display_step'] == 0:
					end = time.time()
					print '{}: lr:{:.7f} D loss:{:.5f}, G loss:{:.5f}, time:{:.3f}'.format(global_step_+CONFIG['train_step_I'], lr_ ,ave_d_loss, ave_g_loss, end-start)
					start = time.time()
					ave_d_loss = 0.0
					ave_g_loss = 0.0

				if global_step_ % CONFIG['save_step'] == 0:
					gen_testmd = []
					gen_trainmd = []
					for k in range(8):
						val_t_h, val_t_e, val_noise = self.data.get_val_data2(k)
						[g_out_] = sess.run([self.stageII['g_out']], feed_dict={self.placeholder['z']:val_noise,
																																	self.placeholder['t_h']:val_t_h,
																																	self.placeholder['t_e']:val_t_e,
																																	self.placeholder['is_train']:False})
						[judge] = sess.run([self.stageII['judge']], feed_dict={self.placeholder['t_h']:val_t_h,
																																	self.placeholder['t_e']:val_t_e,
																																	self.placeholder['x_real96x96']:g_out_,
																																	self.placeholder['is_train']:False})
						min_idx = np.reshape(judge,[-1]).argsort()[:8]
						gen_testmd.append(g_out_[min_idx].copy()+self.data.im_mean)
					gen_testmd = np.reshape(gen_testmd, [-1,96,96,3])
					utils.save_image(gen_testmd, global_step_, os.path.join(CONFIG['ckpt_dir'],'gen_II_bntest'))

					print("save checkpoint_file")
					saver.save(sess,CONFIG['ckpt_dir']+'/model', global_step=global_step_+CONFIG['train_step_I'])

	def generate(self, text_path='sample_testing_text.txt'):
		with open(text_path, 'r') as f:
			lines = [l.replace('\n','') for l in f.readlines()]
		temp_lines = [l.replace(',',' ').replace(' hair','').replace(' eyes','').split(' ') for l in lines]

		idx = [l[0] for l in temp_lines]
		t_h = [self.data.word2idx_h_attr[l[1]] if self.data.word2idx_h_attr.has_key(l[1]) else self.data.word2idx_h_attr['<UNK>'] for l in temp_lines]
		t_e = [self.data.word2idx_e_attr[l[2]] if self.data.word2idx_e_attr.has_key(l[2]) else self.data.word2idx_e_attr['<UNK>'] for l in temp_lines]


		with tf.Session() as sess:
			saver = tf.train.Saver(max_to_keep=3)
			ckpt = tf.train.get_checkpoint_state(CONFIG['ckpt_dir'])
			for n in range(len(ckpt.all_model_checkpoint_paths)):
				ckpt_path = ckpt.all_model_checkpoint_paths[n]
				if ckpt and ckpt_path:
					print "restore {} ...".format(ckpt_path)
					iters = ckpt_path.split('-')[-1]
					saver.restore(sess, ckpt_path)
				else:
					sys.exit(1)

				for i in range(len(idx)):
					noise_temp = np.random.normal(0,STD,[CONFIG['batch_size'],CONFIG['z_dim']])
					t_h_temp = [t_h[i]] * CONFIG['batch_size']
					t_e_temp = [t_e[i]] * CONFIG['batch_size']

					[g_out_] = sess.run([self.stageII['g_out']], feed_dict={self.placeholder['z']:noise_temp,
																																self.placeholder['t_h']:t_h_temp,
																																self.placeholder['t_e']:t_e_temp,
																																self.placeholder['is_train']:False})
					[judge] = sess.run([self.stageII['judge']], feed_dict={self.placeholder['t_h']:t_h_temp,
																																self.placeholder['t_e']:t_e_temp,
																																self.placeholder['x_real96x96']:g_out_,
																																self.placeholder['is_train']:False})
					min_idx = np.reshape(judge,[-1]).argsort()
					image_sort = g_out_[min_idx].copy()+self.data.im_mean
					utils.save_image(image_sort, idx[i], os.path.join(CONFIG['ckpt_dir'],'gen_II_{}_{}'.format(STD, iters)))

					out_sample = random.sample(image_sort, 1)[0]
					out_sample = scipy.misc.imresize(out_sample,(64,64))
					scipy.misc.imsave('samples/sample_{}_{}.jpg'.format(idx[i], n),out_sample)
