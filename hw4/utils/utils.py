import os, sys
import scipy.misc
import numpy as np

def make_ckpt_and_record(dir, info):
	if not os.path.exists(dir):
		os.mkdir(dir)
	with open(os.path.join(dir,'config.txt'), 'w') as f:
		for key, value in info.items():
			f.write('%s:%s\n' % (key, value))


