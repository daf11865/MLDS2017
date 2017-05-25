import os, sys
import scipy.misc
import time
import argparse
import numpy as np
import tensorflow as tf
import pprint
sys.path.append('utils')
import face_data
import ConStackLSGANMD

Model = [ConStackLSGANMD][0]


def parse_args():
	parser = argparse.ArgumentParser(description='annime-GAN')
	parser.add_argument('--testing_text', dest='testing_text',
		                  help='testing text txt file', default = 'sample_testing_text.txt',
		                  type=str)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	gan = Model.gan()
	gan.build_model()
	gan.generate(args.testing_text)
