import collections
import random
import numpy as np
import gensim
import nltk
import sys, os
import pprint

import math
import string
from string import maketrans

from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import*

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

TEST_X = ["hi, what's your name?", 
					"where are you from?", 
					"You should move away from my sight",
					"I think you are bumb. what do you think?",
					"Do you love me?"]

def myTfIdf(sents_tokenized):
	sents_tok_set = []
	for i, sent_tok in enumerate(sents_tokenized):
		sents_tok_set.append(set(sent_tok))

	tokens = [tok for sent_tok in sents_tok_set for tok in sent_tok]
	num_document = float(len(sents_tokenized))
	idf = {word:np.log(num_document/cnt) for word, cnt in collections.Counter(tokens).most_common()}

	sents_tok_score = [[1.0/len(sent_tok) * idf[tok] for tok in sent_tok] if len(sent_tok)>0 else [0.2] for sent_tok in sents_tokenized] # Contribution of same words equals to one word 
	return sents_tok_score
	
class data_loader:
	'''
	i.	Process text data and pre-trained embedding model
	ii.	Form indexed batch data
	iii.Conduct subsampling (similar to tf-idf)
	'''
	def __init__(self, txtfile='data/opensub_glove.txt', batch_size=64, pretrain_ebd_file='glove', subsample_rate=1., voc_size=60000, tfidf=False):
		self.ave_tfidf = tfidf
		self.subsample_rate = subsample_rate
		self.name = txtfile.split('/')[-1].split('.')[0]
		self.voc_size = voc_size

		ebd_model, self.training_text = self.process_data_ebd(txtfile, pretrain_ebd_file)
		self.word2idx = {w:ebd_model[w]['idx'] for w in ebd_model}
		self.idx2word = {ebd_model[w]['idx']:w for w in ebd_model}
		self.voc_cnt = {w:ebd_model[w]['cnt'] for w in ebd_model}

		ebd_size = len(ebd_model['the']['ebd'])
		self.embedding = np.zeros([voc_size,ebd_size])
		for idx in self.idx2word:
			self.embedding[idx] = ebd_model[self.idx2word[idx]]['ebd'][:]

		# subsample words (may not needed)
		def prob_func(frac, subsample_rate):
			return (np.sqrt(frac/subsample_rate)+1)*(subsample_rate/(frac+0.0000000001)) if (np.sqrt(frac/subsample_rate)+1)*(subsample_rate/(frac+0.0000000001)) < 1.0 else 1.0
		total_words = float(sum([cnt for v,cnt in self.voc_cnt.iteritems()]))
		self.voc_weight = {ebd_model[w]['idx']:prob_func(self.voc_cnt[w]/total_words, subsample_rate) for w in ebd_model}
		self.voc_weight[self.word2idx['<PAD>']] = 0.
		self.voc_weight[self.word2idx['<UNK>']] = 0.1

		# tfidf
		self.training_text_weight = self.AveTfIdf([" ".join(A) for Q, A in self.training_text])

		self.training_data = []
		for i,text in enumerate(self.training_text):
			indexed = ([self.word2idx[w] if self.word2idx.has_key(w) else 3 for w in text[0]], [self.word2idx[w] if self.word2idx.has_key(w) else 3 for w in text[1]])
			weight = self.training_text_weight[i]
			length = len(text[1])
			self.training_data.append({'idx':indexed, 'weight':weight, 'length':length, 'text':text}) # list of (Q,A)

		random.shuffle(self.training_data)
		self.i = 0

		#TODO:bucketize
		bucket = [5, 10, 15, 20, 25, 30]
		self.training_data_bucket = []
		for b in bucket:
			self.training_data_bucket.append([sample for sample in self.training_data if sample['length']>b-5 and sample['length']<=b])

		self.training_data_batch = []
		for bucket_data in self.training_data_bucket:
			random.shuffle(bucket_data)
			num_batch = len(bucket_data)/batch_size
			for n in range(num_batch):
				self.training_data_batch.append(bucket_data[n*batch_size:(n+1)*batch_size])

		random.shuffle(self.training_data_batch)

	def get_training_batch(self):
		bs = len(self.training_data_batch[0])
		#if (self.i+1)*bs > len(self.training_data):
		#	random.shuffle(self.training_data)
		#	self.i = 0
		if self.i == len(self.training_data_batch):
			random.shuffle(self.training_data_batch)
			self.i = 0

		batch = self.training_data_batch[self.i] #self.training_data[self.i*bs:(self.i+1)*bs]

		x_ = [[1]+sample['idx'][0]+[2] for sample in batch]
		x_max_len = max([len(x) for x in x_])
		x_padded = [x+[0]*(x_max_len-len(x)) for x in x_]

		y_ = [[1]+sample['idx'][1]+[2] for sample in batch]
		y_max_len = max([len(y) for y in y_])
		y_padded = [y+[0]*(y_max_len-len(y)) for y in y_]

		w_padded = [[1. if idx != 0 else 0. for idx in y] for y in y_padded]
		if self.subsample_rate != 1.:
			w_padded = [[self.voc_weight[idx] for idx in y]+[0.]*(y_max_len-len(y)) for y in y_]
		if self.ave_tfidf:
			w_tfidf_sum = sum([sample['weight'] for sample in batch])
			w_tfidf = np.zeros(np.shape(y_padded), dtype = np.float)
			for n in range(bs):
				w_tfidf[n,:len(y_[n])] = bs*batch[n]['weight']/w_tfidf_sum 
			w_padded = w_padded * w_tfidf

		self.i += 1
		return x_padded, y_padded, w_padded

	def get_val_x(self):
		x_lower = [x.lower() for x in TEST_X]
		x_tok = [nltk.word_tokenize(x) for x in x_lower]
		x_idx = [[self.word2idx[t] if self.word2idx.has_key(t) else self.word2idx['<UNK>'] for t in xt] for xt in x_tok]
		x_ = [[1]+x+[2] for x in x_idx]
		x_max_len = max([len(x) for x in x_])
		x_padded = [x+[0]*(x_max_len-len(x)) for x in x_]
		return x_padded

	def get_x(self, string):
		x = nltk.word_tokenize(string.lower())
		x = [self.word2idx[tok] if self.word2idx.has_key(tok) else self.word2idx['<UNK>'] for tok in x]
		x = [1]+x+[2]
		return [x]

	def process_data_ebd(self, txtfile, pretrain_ebd_file):
		''' current alternative is to eliminate all sentences containing unkwown word
		according to selected vocabulary size.
		'''
		if not os.path.exists(txtfile):
			print "run 'python data/xxx_process.py' first"
			sys.exit(1)
		else:
			with open(txtfile, 'r') as f:
				lines = [l.replace('\n','') for l in f.readlines()]
			sent_tok = [l.split(' ') for l in lines]

			if not os.path.exists('temp'):
				os.mkdir('temp')

			if os.path.exists('temp/{}_{}_dict.npy'.format(self.name, self.voc_size)):
				print "Load {}_{}_dict.npy ...".format(self.name, self.voc_size),
				ebd_model = np.load('temp/{}_{}_dict.npy'.format(self.name, self.voc_size)).item()
			else:
				print "Process {}_{}_dict.npy ...".format(self.name, self.voc_size),
				ebd_model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_ebd_file, binary=True)
				ebdsize = len(ebd_model['the'])

				tokens = [t for st in sent_tok for t in st]
				voc_count = [(v,cnt) for v,cnt in collections.Counter(tokens).most_common() if ebd_model.vocab.has_key(v)][:self.voc_size-4]
				print "Least frequent words:{}".format(voc_count[-20:])
				assert self.voc_size-4 <= len(voc_count), "Maximum voc_size is {}".format(len(voc_count)) 
				voc_count_dict = {v:cnt for v, cnt in voc_count}
				voc_count_dict.update({'<PAD>':0,'<GO>':0,'<EOS>':0,'<UNK>':0})

				idx2word = ['<PAD>','<GO>','<EOS>','<UNK>'] + [v for v,cnt in voc_count]
				word2idx = {idx2word[i]:i for i in range(len(idx2word))}

				pretrain_ebd = np.vstack([np.random.normal(0,0.01,[4,ebdsize]), [ebd_model[w] for w in idx2word[4:]]])
				pretrain_ebd_dict = {idx2word[i]:pretrain_ebd[i] for i in range(len(idx2word))}

				ebd_model = {idx2word[i]: {'idx':i,'cnt':voc_count_dict[idx2word[i]],'ebd':pretrain_ebd_dict[idx2word[i]]} for i in range(len(idx2word))}
				np.save('temp/{}_{}_dict.npy'.format(self.name, self.voc_size), ebd_model)

			# Remove sentence that has more than two unknown words
			sent_tok_pair = [(sent_tok[i],sent_tok[i+1]) for i in xrange(0,len(sent_tok),2)] # list of (Q,A)
			sent_tok_return = []
			for stp in sent_tok_pair:
				keep_cnt = 0
				for w in stp[0]+stp[1]:
					if not ebd_model.has_key(w):
						keep_cnt += 1
				if keep_cnt<3:
					sent_tok_return.append(stp)

			print "{} sample pairs".format(len(sent_tok_return))
		return ebd_model, sent_tok_return

	def AveTfIdf(self, text):
		if os.path.exists('temp/{}_{}_senttfidf.npy'.format(self.name, self.voc_size)):
			print "Load {}_{}_senttfidf.npy ...".format(self.name, self.voc_size)
			weight = np.load('temp/{}_{}_senttfidf.npy'.format(self.name, self.voc_size))
		else:
			print "Process {}_{}_senttfidf.npy ...".format(self.name, self.voc_size)
			lower = [t.lower() for t in text]
			tran = maketrans(string.punctuation, ' '*len(string.punctuation))
			no_punctuation = [l.translate(tran) for l in lower]
			tokens = [no_p.replace('    ',' ').replace('   ',' ').replace('  ',' ').split(' ') for no_p in no_punctuation]

			sw = [str(w) for w in stopwords.words('english')]
			filtered_tokens = [[w for w in ts if not w in sw and w != ''] for ts in tokens]

			#TODO:custom tfidf needed to prevent memory error
			scores = myTfIdf(filtered_tokens)
			weight = [np.sum(scores[i])/len(no_punctuation[i]) for i in range(len(scores))]
			self.sc = scores

			np.save('temp/{}_{}_senttfidf.npy'.format(self.name, self.voc_size), weight)
		return weight

#dl = data_loader(txtfile='data/opensub_glove.txt', subsample_rate=0.001, voc_size=30000)
#x, y, w = dl.get_training_batch()


