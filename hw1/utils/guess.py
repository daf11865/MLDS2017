import numpy as np


def BiRNN_guess_neighbor(sess, model_Output, X_id, Seq_len, Drop_keep, x_id, blank_pos, candidate_id, window_size = 1):
	''' 
	Example.
			x_id: 					[512 ,13333 ,5 ,0 ,99, 4] where 0(<UNK>) is the blank we tend to predict
			blank_pos:			3
			candidate_id:		[54, 2, 7778, 123, 8130]
	We will look at neighbor's probability by placing each of cadidates at the blank
	Note that scores are 2 element less (the frontend and the backend)
	'''
	seq_len = [len(x_id)]
	scores = sess.run([model_Output],feed_dict={X_id:[x_id],
																					 Seq_len: seq_len,
																					 Drop_keep: 1.0})[0]

	candidate_sc = np.log(scores[blank_pos-1][candidate_id])

	if window_size > 0:
		for i in range(len(candidate_id)):
			cid = candidate_id[i]
			x_id[blank_pos] = cid

			scores = sess.run([model_Output],feed_dict={X_id:[x_id],
																				 Seq_len: seq_len,
																				 Drop_keep: 1.0})[0]

			for neighbor in range(1, window_size+1):
				if blank_pos-1 + neighbor < len(scores):
					neighbor_id = x_id[blank_pos + neighbor]
					neighbor_sc = np.log(scores[blank_pos-1 + neighbor][neighbor_id])
					candidate_sc[i] += neighbor_sc
				if blank_pos-1 - neighbor >= 0:
					neighbor_id = x_id[blank_pos - neighbor]
					neighbor_sc = np.log(scores[blank_pos-1 - neighbor][neighbor_id])
					candidate_sc[i] += neighbor_sc

	return candidate_sc/(window_size*2+1)


def UniRNN_guess_neighbor(sess, model_Output, X_id, Seq_len, Drop_keep, x_id, blank_pos, candidate_id, window_size = 1):
	''' 
	Example.
			x_id: 					[512 ,13333 ,5 ,0 ,99, 4] where 0(<UNK>) is the blank we tend to predict
			blank_pos:			3
			candidate_id:		[54, 2, 7778, 123, 8130]
	We will look at neighbor's probability by placing each of cadidates at the blank
	Note that scores are 1 element less (the backend prediction is ignored)
	'''
	seq_len = [len(x_id)]
	scores = sess.run([model_Output],feed_dict={X_id:[x_id],
																					 Seq_len: seq_len,
																					 Drop_keep: 1.0})[0]

	candidate_sc = np.log(scores[blank_pos][candidate_id])

	if window_size > 0:
		for i in range(len(candidate_id)):
			cid = candidate_id[i]
			x_id[blank_pos] = cid

			scores = sess.run([model_Output],feed_dict={X_id:[x_id],
																				 Seq_len: seq_len,
																				 Drop_keep: 1.0})[0]

			for neighbor in range(1, window_size+1):
				if blank_pos + neighbor < len(scores):
					neighbor_id = x_id[blank_pos + neighbor]
					neighbor_sc = np.log(scores[blank_pos + neighbor][neighbor_id])
					candidate_sc[i] += neighbor_sc

	return candidate_sc/(window_size+1)



# Something wrong?
def BiRNN_Hierarchical_guess_neighbor(sess, model_Output, X_id, Seq_len, Drop_keep, x_id, blank_pos, candidate_id, tree, window_size = 1, average = False):
	def prob(logits,idx,tree,average):
		def sigmoid(x):
			return 1.0/(1.0+np.exp(x))
		lg = logits[np.array(tree.idx2nodes[idx])]
		lg2sigmoid = [sigmoid(l) for l in lg_r]
		code = tree.idx2code[idx]
		probs = [lg2sigmoid[i] if code[i] == '1' else 1-lg2sigmoid[i] for i in range(len(lg))]
		#logprobs = np.array([log_sigmoid(lg[i],1.0) if code[i]=='1' else log_sigmoid(lg[i],0.0) for i in range(len(lg))])
		logprobs = [np.log(prob) for prob in probs]
		assert not np.any(np.isinf(logprobs)),'probs:{}\nlogprobs:{}'.format(probs,logprobs)
		if average:
			logprob = np.mean(logprobs)
		else:
			logprob = np.sum(logprobs)
		prob = np.exp(logprob)
		return prob

	def score(logits,idx,tree,average):
		def sigmoid_crossenthropy(x,z):
			return max(x, 0) - x * z + np.log(1 + np.exp(-abs(x)))
		lg = logits[np.array(tree.idx2nodes[idx])]
		code = tree.idx2code[idx]
		losses = [sigmoid_crossenthropy(lg[i],int(code[i])) for i in range(len(lg))]
		loss = np.mean(losses) if average else np.sum(losses)
		return -loss

	seq_len = [len(x_id)]
	logits = sess.run([model_Output],feed_dict={X_id:[x_id],
																					 		Seq_len: seq_len,
																							Drop_keep: 1.0})[0]
	logit_blank = logits[blank_pos-1]
	#candidate_probs = [prob(logit_blank,cid,tree,average) for cid in candidate_id]
	#candidate_score = np.log(candidate_probs)
	candidate_score = np.array([score(logit_blank,cid,tree,average) for cid in candidate_id])

	if window_size > 0:
		for i in range(len(candidate_id)):
			cid = candidate_id[i]
			x_id[blank_pos] = cid

			logits = sess.run([model_Output],feed_dict={X_id:[x_id],
																				 Seq_len: seq_len})[0]

			for neighbor in range(1, window_size+1):
				if blank_pos-1 + neighbor < len(logits):
					neighbor_id = x_id[blank_pos + neighbor]
					logit_neighbor = logits[blank_pos-1 + neighbor]
					#neighbor_prob = prob(logit_neighbor,neighbor_id,tree,average)
					#neighbor_sc = np.log(neighbor_prob)
					neighbor_score = score(logit_neighbor,neighbor_id,tree,average)
					candidate_score[i] += neighbor_score
				if blank_pos-1 - neighbor >= 0:
					neighbor_id = x_id[blank_pos - neighbor]
					logit_neighbor = logits[blank_pos-1 - neighbor]
					#neighbor_prob = prob(logit_neighbor,neighbor_id,tree,average)
					#neighbor_sc = np.log(neighbor_prob)
					neighbor_score = score(logit_neighbor,neighbor_id,tree,average)
					candidate_score[i] += neighbor_score

	return candidate_score/(window_size*2+1)
