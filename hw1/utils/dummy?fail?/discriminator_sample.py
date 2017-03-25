import numpy as np
import heapq

def get_batch(x_id_padded, seq_len, sentence_num, pred_prob):
	sent_cand = np.random.choice(range(len(x_id_padded)), sentence_num, replace=False)

	pred_prob_reshape = []
	count = 0
	for i in range(len(seq_len)):
		sent_per_word_probs = pred_prob[count:count+seq_len[i]-2]
		pred_prob_reshape.append(sent_per_word_probs)
		count = seq_len[i]-2

	negtive_sample_num = len(x_id_padded)/sentence_num - 1 # 3 if sentence_num = 16 and batch_size = 64

	x_id_cand = []
	y_id_cand = []
	seq_len_cand = []
	cand_id = []
	batch_size_count = 0
	for cand in sent_cand:
		this_sent_pred = pred_prob_reshape[cand] # [pred_probs[0],pred_probs[1],...,pred_probs(k-1)]
		this_sent = x_id_padded[cand] # [id('I'),id('do'),id('not'),id('often'),id('go'),id('to'),id('library'),id('.'),id(<PAD>),...,id(<PAD>)]
		this_sent_len = seq_len[cand] # k
		target_word_pos = np.random.randint(1,this_sent_len-1) # one number in 1 ~ k-1, let's say, 3

		target_word = this_sent[target_word_pos] # id('often')
		target_word_predictions = this_sent_pred[target_word_pos-1] # predicted results of Voc_Size words, where answer is id('often')
		pred_word_topN = np.argpartition(-target_word_predictions,negtive_sample_num+1)[:negtive_sample_num+1] # id('cat'), id('watch'), id('often'), id('fresh')
		pred_negative_word_topN = [negative_word for negative_word in pred_word_topN if negative_word != target_word] #  id('cat'), id('watch'), id('fresh')

		x_id_cand.append(this_sent[:])
		y_id_cand.append([1])
		seq_len_cand.append(this_sent_len)
		cand_id.append([batch_size_count,target_word_pos])
		batch_size_count +=1
		for neg_word in pred_negative_word_topN:
			this_sent_temp = this_sent[:]
			this_sent_temp[target_word_pos] = neg_word

			x_id_cand.append(this_sent_temp[:])
			y_id_cand.append([0])
			seq_len_cand.append(this_sent_len)
			cand_id.append([batch_size_count,target_word_pos])
			batch_size_count +=1

	return x_id_cand, y_id_cand, seq_len_cand, cand_id
