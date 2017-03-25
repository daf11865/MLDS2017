'''Deprecated, not in use'''
import sys, os
import re
import collections
import nltk
from nltk.tokenize import RegexpTokenizer
import random
import numpy as np
import pprint
import re, string

Training_Data_Dir = 'Holmes_Training_Data'
Val_Data_Dir = 'Holmes_Val_Data'

def process_text(text):
	'''Return: list of lists(sentences) of words, punctuations, and logits'''
	text_words = nltk.word_tokenize(text)
	text_sents = nltk.sent_tokenize(" ".join(text_words))
	replace_list = {" '":" ' ", ":":",", "--":",", "!":".", "?":".", ";":".", "...":"."}
	for key in replace_list.keys():
		text_sents = [s.replace(key,replace_list[key]) for s in text_sents]
	text_sents = [nltk.word_tokenize(s) for s in text_sents]
	text_sents = [' '.join(s).lower() for s in text_sents]
	return text_sents

def process_raw_data():
	raw_data_sents = []
	dataset_files = os.listdir(Training_Data_Dir)
	for file in dataset_files[:5]:
		with open(os.path.join(Training_Data_Dir,file),'r') as f:
			print("Processing {}".format(file))
			text = f.read().split("*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END*")[-1]
			text = text[50:-50]
			text_sents = process_text(text)
		raw_data_sents.extend(text_sents)

	val_raw_data_sents = []
	val_dataset_files = os.listdir(Val_Data_Dir)
	for file in val_dataset_files:
		with open(os.path.join(Val_Data_Dir,file),'r') as f:
			print("Processing {}".format(file))
			val_text = f.read().split("*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END*")[-1]
			val_text = val_text[50:-50]
			val_text_sents = process_text(val_text)
		val_raw_data_sents.extend(val_text_sents)

	with open('raw_training_data.txt','w') as fw:
		for s in raw_data_sents:
			fw.write(s+'\n')

	with open('raw_val_data.txt','w') as fw:
		for s in val_raw_data_sents:
			fw.write(s+'\n')

if __name__ == "__main__":
	process_raw_data()
