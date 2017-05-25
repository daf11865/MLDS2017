import csv
import os, sys
import numpy as np
import skimage
import skimage.io
import skimage.transform
import scipy.misc
import random
from matplotlib import pyplot as plt
#sys.path.append('skip-thoughts')
#import skipthoughts
from skimage import exposure
import pprint

TAG_PATH = 'tags_clean.csv'
VAL_TAGS = ['black hair blue eyes', 'pink hair green eyes', 'green hair green eyes', 'blue hair red eyes', 
						'red hair gray eyes', 'blue hair green eyes', 'blonde hair yellow eyes', 'green hair blue eyes']
print VAL_TAGS

def flip_batch(ims):
	return np.array([im[:,::-1,:] if random.randint(0,1) else im for im in ims])

class data():
	def __init__(self, config):
		self.config = config
		
		print('1) Load images ...')
		ims = np.load('temp/ims.npy')
		ims48x48 = np.load('temp/ims48x48.npy')
		ims64x64 = np.load('temp/ims64x64.npy')

		print('2) Load tags ...')
		tags = np.load('temp/tags.npy')

		print('3) Load dictionary ...')
		dictionary = np.load('temp/dictionary.npy')
		self.word2idx_h_attr = {dictionary[0][i]:i for i in range(len(dictionary[0]))}
		self.word2idx_e_attr = {dictionary[1][i]:i for i in range(len(dictionary[1]))}
		pprint.pprint(dictionary)
		# Process
		tags_indexed = [None for i in range(len(tags))]
		for i in range(len(tags)):
			if len(tags[i]) > 0 :
				tags_indexed[i] = self.tag2idx(tags[i])
			else: 
				tags_indexed[i] = [(self.word2idx_h_attr['<UNK>'],self.word2idx_e_attr['<UNK>'])]

		if self.config['use_tagless_data']:
			self.data = [{'image':ims[i], 'image48x48':ims48x48[i], 'image64x64':ims64x64[i], 'tag':tags[i], 'tag_indexed':tags_indexed[i]} for i in range(len(ims))] 
		else:
			self.data = [{'image':ims[i], 'image48x48':ims48x48[i], 'image64x64':ims64x64[i], 'tag':tags[i], 'tag_indexed':tags_indexed[i]} for i in range(len(ims)) if len(tags[i])>0]
		self.data_all = [{'image':ims[i], 'image48x48':ims48x48[i], 'image64x64':ims64x64[i], 'tag':tags[i], 'tag_indexed':tags_indexed[i]} for i in range(len(ims))]
	
		self.im_mean = np.mean([d['image'] for d in self.data], (0,1,2)) / 255.0 if self.config['sub_mean'] else 0.5

		self.cnt = 0
		self.training_data = self.data[:]
		random.shuffle(self.training_data)
		print('Total {} training samples'.format(len(self.training_data)))

	def tag2idx(self, tags):
		ti = []
		for t in tags:
			if len(t.split()) == 2 and t.split()[1] == 'hair':
				ti.append((self.word2idx_h_attr[t.split(' ')[0]], self.word2idx_e_attr['<UNK>']))
			elif len(t.split()) == 2 and t.split()[1] == 'eyes':
				ti.append((self.word2idx_h_attr['<UNK>'], self.word2idx_e_attr[t.split(' ')[0]]))
			elif len(t.split()) == 4:
				ti.append((self.word2idx_h_attr[t.split(' ')[0]],self.word2idx_e_attr[t.split(' ')[2]]))
		return ti

	def get_training_batch(self, check=False):
		'''
		Return:
			a batch of attribute index
			a batch of real images
			a batch of wrong images
			a batch of noise samples
		'''
		batch_size = self.config['batch_size']
		if batch_size * (self.cnt+1) > len(self.training_data):
			random.shuffle(self.training_data)
			self.cnt = 0

		batch_data = self.training_data[batch_size * self.cnt:batch_size * (self.cnt+1)]

		batch_tag_indexed = [random.sample(d['tag_indexed'],1)[0] for d in batch_data]

		batch_tag_indexed_h = [t[0]for t in batch_tag_indexed]
		batch_tag_indexed_e = [t[1]for t in batch_tag_indexed]
		batch_noise_sample = np.random.normal(0,1,[batch_size,self.config['z_dim']]) if self.config['z_normal'] else np.random.uniform(-1,1,[batch_size,self.config['z_dim']])
		batch_real_im = np.array([d['image']/255.0-self.im_mean for d in batch_data])
		batch_real_im48x48 = np.array([d['image48x48']/255.0-self.im_mean for d in batch_data])
		batch_real_im64x64 = np.array([d['image64x64']/255.0-self.im_mean for d in batch_data])
		if self.config['sample_tagless_data']:
			batch_wrong_im = np.array([d['image']/255.0-self.im_mean for d in random.sample(self.data_all, batch_size)])
			batch_wrong_im48x48 = np.array([d['image48x48']/255.0-self.im_mean for d in random.sample(self.data_all, batch_size)])
			batch_wrong_im64x64 = np.array([d['image64x64']/255.0-self.im_mean for d in random.sample(self.data_all, batch_size)])
		else:
			batch_wrong_im = np.array([d['image']/255.0-self.im_mean for d in random.sample(self.training_data, batch_size)])
			batch_wrong_im48x48 = np.array([d['image48x48']/255.0-self.im_mean for d in random.sample(self.training_data, batch_size)])
			batch_wrong_im64x64 = np.array([d['image64x64']/255.0-self.im_mean for d in random.sample(self.training_data, batch_size)])
		if self.config['flip']:
			batch_real_im = flip_batch(batch_real_im)
			batch_wrong_im = flip_batch(batch_wrong_im)
			batch_real_im48x48 = flip_batch(batch_real_im48x48)
			batch_wrong_im48x48 = flip_batch(batch_wrong_im48x48)
			batch_real_im64x64 = flip_batch(batch_real_im64x64)
			batch_wrong_im64x64 = flip_batch(batch_wrong_im64x64)

		self.cnt += 1			

		if check:
			for i in range(len(batch_data)):
				print batch_data[i]['tag']
				print batch_data[i]['tag_indexed']

				skimage.io.imshow((batch_real_im[i]+1)/2.0,)
				plt.show()
				skimage.io.imshow((batch_wrong_im[i]+1)/2.0,)
				plt.show()

		dic = { 't_h':batch_tag_indexed_h ,
						't_e':batch_tag_indexed_e, 
						'x_real96x96':batch_real_im, 
						'x_real48x48': batch_real_im48x48, 
						'x_real64x64':batch_real_im64x64, 
						'x_wrong96x96':batch_wrong_im, 
						'x_wrong48x48':batch_wrong_im48x48, 
						'x_wrong64x64':batch_wrong_im64x64, 'z':batch_noise_sample}
		return dic

	def get_val_data(self):
		n = self.config['batch_size'] / len(VAL_TAGS)
		val_noise_sample = np.random.normal(0,1,[self.config['batch_size'],self.config['z_dim']])
		val_tag_indexed_h = [attr[0] for attr in self.tag2idx(VAL_TAGS) for i in range(n)]
		val_tag_indexed_e = [attr[1] for attr in self.tag2idx(VAL_TAGS) for i in range(n)]

		return val_tag_indexed_h, val_tag_indexed_e, val_noise_sample

	def get_val_data2(self, k):
		val_noise_sample = np.random.normal(0,1,[self.config['batch_size'],self.config['z_dim']])
		val_tag_indexed_h = [attr[0] for attr in self.tag2idx([VAL_TAGS[k]]) for i in range(self.config['batch_size'])]
		val_tag_indexed_e = [attr[1] for attr in self.tag2idx([VAL_TAGS[k]]) for i in range(self.config['batch_size'])]

		return val_tag_indexed_h, val_tag_indexed_e, val_noise_sample

if __name__ == "__main__":
	d = data({'batch_size':64,'simple_tags':True,'z_dim':100,'use_tagless_data':False,'flip':True,'x_real_noise':True,'z_normal':True,'sample_tagless_data':False})
	td = d.get_training_batch(check=True)
