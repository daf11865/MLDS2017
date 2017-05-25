import os, sys
import scipy.misc
import numpy as np

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
