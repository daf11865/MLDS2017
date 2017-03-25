import sys, os
import re
import collections
import nltk
from nltk.tokenize import RegexpTokenizer
import random
import numpy as np
import pprint
import re, string

class Holmes_Training_Data(object):
	def __init__(self, training_data_file = 'train.txt', val_data_file = 'val.txt', clip_num = 60):
		self.raw_training_data = None
		self.raw_val_data = None
		self.raw_word_count = None
		self.raw_word_list = None

		self.word2idx = None
		self.training_data = None
		self.val_data = None
		self.wordid_count = None
		self.wordid_keep_loss_prob = None

		self.step_count = 0

		with open(training_data_file,'r') as tf:
			raw_training_text = tf.read().splitlines()
			#self.raw_training_data = [s.lower().split() for s in raw_training_text if len(s.split())<60]
			self.raw_training_data = [s.split() for s in raw_training_text]
		with open(val_data_file,'r') as vf:
			raw_val_text = vf.read().splitlines()
			#self.raw_val_data = [s.lower().split() for s in raw_val_text if len(s.split())<60]
			self.raw_val_data = [s.split() for s in raw_val_text]
		corpus = [w for s in self.raw_training_data for w in s]
		self.raw_word_count = collections.Counter(corpus).most_common()
		self.raw_word_list = [w_c[0] for w_c in self.raw_word_count]
		random.shuffle(self.raw_training_data)
		random.shuffle(self.raw_val_data)
		'''with open('train.txt','w') as wf:
			for s in self.raw_training_data:
				wf.write(' '.join(s)+'\n') 
		with open('val.txt','w') as wf:
			for s in self.raw_val_data:
				wf.write(' '.join(s)+'\n')'''
		print('Set up raw data: {} train sentence {} val sentence'.format(len(self.raw_training_data),len(self.raw_val_data)))

	def build_data(self, word2idx, sample_rate):
		self.word2idx = word2idx

		def sentence2word2idx(sent,word2idx):
			return [word2idx[w] if word2idx.has_key(w) else word2idx['<UNK>'] for w in s]
		self.training_data = [sentence2word2idx(s,word2idx) for s in self.raw_training_data]
		self.val_data = [sentence2word2idx(s,word2idx) for s in self.raw_val_data]
		random.shuffle(self.training_data)
		random.shuffle(self.val_data)

		def prob_func(frac, sample_rate):
			return (np.sqrt(frac/sample_rate)+1)*(sample_rate/(frac+0.0000000001)) if (np.sqrt(frac/sample_rate)+1)*(sample_rate/(frac+0.0000000001)) < 1.0 else 1.0
		training_data_flat = [wid for s in self.training_data for wid in s]
		wordid_count_temp = collections.Counter(training_data_flat).most_common()
		self.wordid_count = [0] * len(self.word2idx)
		for idx, count in wordid_count_temp:
			self.wordid_count[idx] = count
		#self.wordid_keep_loss_prob = [prob_func(float(count)/len(training_data_flat),sample_rate=sample_rate) for count in self.wordid_count[:-1]] # Ignore <PAD>
		self.wordid_keep_loss_prob = [prob_func(float(count)/len(training_data_flat),sample_rate=sample_rate) for count in self.wordid_count]
		self.wordid_keep_loss_prob[0] = self.wordid_keep_loss_prob[-1] = 0.0 # Set weights of loss of <PAD>, <UNK> to 0.0

		print('Set up HTD.training_data, HTD.val_data, HTD.wordid_keep_loss_prob')

	def get_batch(self, batch_size=32):
		if batch_size * (self.step_count+1) > len(self.training_data):
			self.step_count = 0

		inputs = labels = self.training_data[batch_size * self.step_count: batch_size *(self.step_count+1)]
		seq_len = [len(s) for s in inputs] 
		max_seq_len = max(seq_len)

		inputs_padded = labels_padded = [s + [self.word2idx['<PAD>']]*(max_seq_len-len(s)) for s in inputs]
		# Multiply loss weight and mask out loss of <PAD>
		#mask = [[0.0 if wid == self.word2idx['<PAD>'] or wid == self.word2idx['<UNK>'] else self.wordid_keep_loss_prob[wid] for wid in s] for s in labels_padded ] 
		mask = [[self.wordid_keep_loss_prob[wid] for wid in s] for s in labels_padded ] 

		self.step_count += 1
		return inputs_padded, labels_padded, seq_len, mask


	def get_val_batch(self,batch_size=32,step=None):
		if step:
			inputs = labels = self.val_data[step*batch_size:(step+1)*batch_size]
		else:
			random.shuffle(self.val_data)
			inputs = labels = self.val_data[0:batch_size]
		seq_len = [len(s) for s in inputs] 
		max_seq_len = max(seq_len)

		inputs_padded = labels_padded = [s + [self.word2idx['<PAD>']]*(max_seq_len-len(s)) for s in inputs]
		#mask = [[0.0 if wid == self.word2idx['<PAD>'] or wid == self.word2idx['<UNK>'] else self.wordid_keep_loss_prob[wid] for wid in s] for s in labels_padded ]
		mask = [[self.wordid_keep_loss_prob[wid] for wid in s] for s in labels_padded ] 
		return inputs_padded, labels_padded, seq_len, mask
