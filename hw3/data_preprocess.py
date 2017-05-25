import sys, os
import numpy as np
import skimage, skimage.transform, skimage.io
import csv
import collections
import pprint

IMAGE_DIR = 'faces'
TAG_PATH = 'tags_clean.csv'
VAL_TAGS = ['black hair blue eyes', 'pink hair green eyes', 'green hair green eyes', 'blue hair red eyes']

if not os.path.exists('temp'):
	os.mkdir('temp')

# Save images, resized images
print('1) Save images, resized images ...')
im_paths = os.listdir(IMAGE_DIR)

if not os.path.exists('temp/ims.npy'):
	ims = [[] for i in range(len(im_paths))]
	for i,p in enumerate(im_paths):
		path = os.path.join(IMAGE_DIR,p)
		im = skimage.io.imread(path)
		idx = int(p.replace('.jpg',''))
		ims[idx] = skimage.img_as_ubyte(im)
		if i % 100 == 0:
			print('{} saved'.format(i))
	np.save('temp/ims',ims)

if not os.path.exists('temp/ims64x64.npy'):
	resized_ims = [[] for i in range(len(im_paths))]
	for i,p in enumerate(im_paths):
		path = os.path.join(IMAGE_DIR,p)
		resized_im = skimage.transform.resize(skimage.io.imread(path),(64,64))
		idx = int(p.replace('.jpg',''))
		resized_ims[idx] = skimage.img_as_ubyte(resized_im)
		if i % 100 == 0:
			print('{} saved'.format(i))
	np.save('temp/ims64x64',resized_ims)

if not os.path.exists('temp/ims48x48.npy'):
	resized_ims = [[] for i in range(len(im_paths))]
	for i,p in enumerate(im_paths):
		path = os.path.join(IMAGE_DIR,p)
		resized_im = skimage.transform.resize(skimage.io.imread(path),(48,48))
		idx = int(p.replace('.jpg',''))
		resized_ims[idx] = skimage.img_as_ubyte(resized_im)
		if i % 100 == 0:
			print('{} saved'.format(i))
	np.save('temp/ims48x48',resized_ims)

# Save simplified tags
print('2) Save tags and dictionary...')
with open(TAG_PATH) as f:
	raw_tags = [l[1].split('\t') for l in csv.reader(f)]

if not os.path.exists('temp/tags.npy'):
	keep_simple_tags = [[string.split(':')[0] for string in rt if ' eyes' in string or ' hair' in string] for rt in raw_tags]
	for i in range(len(keep_simple_tags)):
		keep = []
		for ht in [hair_tag for hair_tag in keep_simple_tags[i] if 'hair' in hair_tag]:
			if 'long' in ht or 'short' in ht or 'pubic' in ht or 'damage' in ht:
				continue
			for et in [eyes_tag for eyes_tag in keep_simple_tags[i] if 'eyes' in eyes_tag]:
				if 'bicolored' in et or '11' in et:
					continue
				keep.append(ht+' '+et)

		for ht in [hair_tag for hair_tag in keep_simple_tags[i] if 'hair' in hair_tag]:
			if 'long' in ht or 'short' in ht or 'pubic' in ht or 'damage' in ht:
				continue
			keep.append(ht)

		for et in [eyes_tag for eyes_tag in keep_simple_tags[i] if 'eyes' in eyes_tag]:
			if 'bicolored' in et or '11' in et:
				continue
			keep.append(et)

		keep_simple_tags[i] = keep
	np.save('temp/tags',keep_simple_tags)

if not os.path.exists('temp/dictionary.npy'):
	keep_simple_tags = [[string.split(':')[0] for string in rt if ' eyes' in string or ' hair' in string] for rt in raw_tags]
	hair_attr = []
	eyes_attr = []
	for i in range(len(keep_simple_tags)):
		for ht in [hair_tag for hair_tag in keep_simple_tags[i] if 'hair' in hair_tag]:
			if 'long' in ht or 'short' in ht or 'pubic' in ht or 'damage' in ht:
				continue
			hair_attr.append(ht.replace(' hair', ''))
		for et in [eyes_tag for eyes_tag in keep_simple_tags[i] if 'eyes' in eyes_tag]:
			if 'bicolored' in et or '11' in et:
				continue
			eyes_attr.append(et.replace(' eyes', ''))

	hair_count = collections.Counter(hair_attr).most_common()
	eyes_count = collections.Counter(eyes_attr).most_common()

	hair_dict = ['<UNK>'] + [tup[0] for tup in hair_count]
	eyes_dict = ['<UNK>'] + [tup[0] for tup in eyes_count]

	np.save('temp/dictionary',[hair_dict,eyes_dict])

	print len(hair_dict)
	pprint.pprint(hair_dict)
	print len(eyes_dict)
	pprint.pprint(eyes_dict)
