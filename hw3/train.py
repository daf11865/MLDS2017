import os, sys
import scipy.misc
import time
import numpy as np
import tensorflow as tf
import pprint
sys.path.append('utils')
import face_data
import ConStackLSGANMD

Model = [ConStackLSGANMD][0]

if __name__ == "__main__":
	gan = Model.gan()
	gan.build_model()
	gan.train()
