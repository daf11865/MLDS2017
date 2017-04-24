import sys, os
import collections
import nltk
import random
import numpy as np
import json
import pickle

class Video_Data_Feat(object):
	def __init__(self, 
						train_feat_dir="MLDS_hw2_data/training_data/feat", 
						train_cap_json="MLDS_hw2_data/training_label.json", 
						test_feat_dir="MLDS_hw2_data/testing_data/feat",
						test_cap_json="MLDS_hw2_data/testing_public_label.json"):

		self.train_data = []
		self.test_data = []
		self.train_data_format2 = []
		self.raw_word_count = None
		self.total_train_sample = 0

		self.word2idx = None
		self.train_cap_id = None
		self.test_cap_id = None

		self.train_step_count = 0

		if os.path.exists('temp/VDF_data.npy'):
			[self.train_data, self.test_data]= np.load('temp/VDF_data.npy')
		else:
			train_feat = {}
			train_cap = {}
			for feat_npy in os.listdir(train_feat_dir):
				train_feat.update({feat_npy.replace('.npy',''):np.load(os.path.join(train_feat_dir,feat_npy))})
			with open(train_cap_json,'r') as f:
				for cap in json.load(f):
					train_cap.update({cap['id']:cap['caption']})
			for idx in train_feat:
				caps = [[w for w in nltk.word_tokenize(str(sent).lower()) if w != "."] for sent in train_cap[idx]]
				self.train_data.append({'feat':train_feat[idx], 'caption':caps})

			test_feat = {}
			test_cap = {}
			for feat_npy in os.listdir(test_feat_dir):
				test_feat.update({feat_npy.replace('.npy',''):np.load(os.path.join(test_feat_dir,feat_npy))})
			with open(test_cap_json,'r') as f:
				for cap in json.load(f):
					test_cap.update({cap['id']:cap['caption']})
			for idx in test_feat:
				caps = [[w for w in nltk.word_tokenize(str(sent).lower()) if w != "."] for sent in test_cap[idx]]
				self.test_data.append({'id':idx, 'feat':test_feat[idx], 'caption':caps})

			np.save('temp/VDF_data.npy',[self.train_data, self.test_data])

		corpus = []
		for d in self.train_data + self.test_data: # Including test words (though never encounter some test words in training time)
			for words in d['caption']:
				corpus += words
		self.raw_word_count = collections.Counter(corpus).most_common()

		for td in self.train_data:
			self.total_train_sample += len(td['caption'])

	def build_data(self, word2idx, sample_rate = 0.001):
		self.word2idx = word2idx # [<PAD>:0,<BOS>:1,<EOS>:2,<UNK>:3,'a':4,'the':5,...]
		self.idx2word = {v: k for k, v in word2idx.iteritems()}
		for train_sample in self.train_data:
				temp = [[word2idx[w] if word2idx.has_key(w) else word2idx['<UNK>'] for w in sent] for sent in train_sample['caption']]
				train_sample['caption_id'] = []
				for cap in temp:
					if cap not in train_sample['caption_id']:
						train_sample['caption_id'].append(cap)
						self.train_data_format2.append({"caption_id":cap, "feat":train_sample['feat']})
		for test_sample in self.test_data:
				temp = [[word2idx[w] if word2idx.has_key(w) else word2idx['<UNK>'] for w in sent] for sent in test_sample['caption']]
				test_sample['caption_id'] = []
				for cap in temp:
					if cap not in test_sample['caption_id']:
						test_sample['caption_id'].append(cap)
		random.shuffle(self.train_data)
		random.shuffle(self.train_data_format2)

		all_wordid = [wid for td in self.train_data for sentid in td['caption_id'] for wid in sentid]
		wordid_count_temp = collections.Counter(all_wordid).most_common()
		self.wordid_count = [0] * len(self.word2idx)
		for idx, count in wordid_count_temp:
			self.wordid_count[idx] = count

		def prob_func(frac, sample_rate):
			return (np.sqrt(frac/sample_rate)+1)*(sample_rate/(frac+0.0000000001)) if (np.sqrt(frac/sample_rate)+1)*(sample_rate/(frac+0.0000000001)) < 1.0 else 1.0
		self.wordid_keep_loss_prob = [prob_func(float(count)/len(all_wordid),sample_rate=sample_rate) for count in self.wordid_count]
		self.wordid_keep_loss_prob[0] = 0.0
		self.wordid_keep_loss_prob[2] = self.wordid_keep_loss_prob[4]

		print("Index-Word: Count SubsamplingRate")
		for idx in range(10):
			print("{}-{}: {} {}".format(idx,self.idx2word[idx],self.wordid_count[idx],self.wordid_keep_loss_prob[idx]))

	def get_train_batch(self, batch_size=32, seq_reward = 1.0):
		if batch_size * (self.train_step_count+1) > len(self.train_data_format2):
			random.shuffle(self.train_data_format2)
			self.train_step_count = 0

		start = batch_size * self.train_step_count
		end = 	batch_size * (self.train_step_count+1)
		v_feat = np.array([train_sample['feat'] for train_sample in self.train_data_format2[start:end]])

		c_idx = [train_sample['caption_id'] for train_sample in self.train_data_format2[start:end]]
		c_idx_ = [[self.word2idx['<BOS>']] + cap for cap in c_idx]
		l_idx_ = [cap + [self.word2idx['<EOS>']] for cap in c_idx]

		v_frames_num = len(v_feat[0]) # 80
		c_idx_len = np.array([len(sent) for sent in c_idx_])

		max_c_idx_len = max(c_idx_len)
		c_idx_padded = np.array([sent+[self.word2idx['<PAD>']]*(max_c_idx_len-len(sent)) for sent in c_idx_])
		l_idx_padded = np.array([sent+[self.word2idx['<PAD>']]*(max_c_idx_len-len(sent)) for sent in l_idx_])

		loss_mask = np.array([[self.wordid_keep_loss_prob[wid_this_sent[i]]*(seq_reward**i)  for i in range(len(wid_this_sent))] for wid_this_sent in l_idx_padded])

		self.train_step_count += 1
		return v_feat, c_idx_padded, l_idx_padded, c_idx_len, loss_mask

	def get_test_batch(self, batch_size = 1, nth = 0):
		video_inputs =[sample['feat'] for sample in self.test_data[nth*batch_size:(nth+1)*batch_size]]
		label = [[" ".join(c) for c in sample['caption']] for sample in self.test_data[nth*batch_size:(nth+1)*batch_size]]
		idx = [sample['id'] for sample in self.test_data[nth*batch_size:(nth+1)*batch_size]]

		return video_inputs, label, idx

	'''def limited_task_data(self, feat_dir = "MLDS_hw2_time_limited/feat"):
		data = []
		for feat_npy in os.listdir(feat_dir):
			data.append({'feat':np.load(os.path.join(feat_dir,feat_npy)), "id":feat_npy.replace('.npy','')})

		return data'''
