import tensorflow as tf

class Beam_Search(object):
	def __init__(self, beam_width = 1):
		self.beam_width = beam_width
		self.beam_gen = tf.constant([[]]*self.beam_width, dtype = tf.int32)
		self.beam_score = tf.zeros([self.beam_width])
		self.i = 0

	def step(self, probs, current_dec_state):
		if self.i == 0:
			current_probs, current_gen = tf.nn.top_k(probs,self.beam_width)

			self.beam_gen = tf.reshape(current_gen,[self.beam_width,1])
			self.beam_score = tf.log(tf.reshape(current_probs,[-1]))

			dec_state = (tf.concat([current_dec_state[0]]*self.beam_width, axis = 0), tf.concat([current_dec_state[1]]*self.beam_width, axis = 0))
			dec_index = self.beam_gen[:,-1]
		else:
			current_probs, current_gen = tf.nn.top_k(probs,self.beam_width)

			previous_gen = tf.stack([self.beam_gen]*self.beam_width, axis = 1)
			previous_gen = tf.reshape(previous_gen,[self.beam_width*self.beam_width,-1])
			current_gen = tf.reshape(current_gen,[self.beam_width*self.beam_width])
			current_gen = tf.where(tf.equal(previous_gen[:,-1],2), tf.fill(tf.shape(current_gen),2), current_gen)
			current_gen = tf.expand_dims(current_gen,1)
			cand_gen_temp = tf.concat([previous_gen,current_gen],axis = 1)

			previous_score = tf.reshape(self.beam_score,[self.beam_width,1])
			previous_score = tf.reshape(tf.concat([previous_score]*self.beam_width, axis = 1), [self.beam_width*self.beam_width])
			current_score = tf.reshape(tf.log(current_probs),[self.beam_width*self.beam_width])
			current_score = tf.where(tf.equal(previous_gen[:,-1],2), tf.fill(tf.shape(current_score),0.0), current_score)
			cand_score_temp = tf.add(previous_score,current_score)

			cand_dec_state_temp = (tf.stack([current_dec_state[0]]*self.beam_width, axis = 1), tf.stack([current_dec_state[1]]*self.beam_width, axis = 1))
			cand_dec_state_temp = (tf.reshape(cand_dec_state_temp[0],[self.beam_width*self.beam_width,-1]),tf.reshape(cand_dec_state_temp[1],[self.beam_width*self.beam_width,-1]))


			EOS_indice = tf.reshape(tf.where(tf.equal(cand_gen_temp[:,-1],2)),[-1])[:1]
			Keep_indice = tf.reshape(tf.where(tf.not_equal(cand_gen_temp[:,-1],2)),[-1])

			EOS_gen = tf.gather(cand_gen_temp, EOS_indice)
			Keep_gen = tf.gather(cand_gen_temp, Keep_indice)
			cand_gen = tf.concat([EOS_gen,Keep_gen],axis = 0)

			EOS_score = tf.gather(cand_score_temp, EOS_indice)
			Keep_score = tf.gather(cand_score_temp, Keep_indice)
			cand_score = tf.concat([EOS_score,Keep_score],axis = 0)

			EOS_dec_state = (tf.gather(cand_dec_state_temp[0], EOS_indice), tf.gather(cand_dec_state_temp[1], EOS_indice))
			Keep_dec_state = (tf.gather(cand_dec_state_temp[0], Keep_indice), tf.gather(cand_dec_state_temp[1], Keep_indice))
			cand_dec_state = (tf.concat([EOS_dec_state[0],Keep_dec_state[0]],axis = 0), tf.concat([EOS_dec_state[1],Keep_dec_state[1]],axis = 0))

			score, indices = tf.nn.top_k(cand_score, self.beam_width)

			self.beam_score = tf.gather(cand_score, indices)
			self.beam_gen = tf.gather(cand_gen, indices)

			if self.beam_width == 1:
				dec_state = current_dec_state
			else:
				dec_state = (tf.gather(cand_dec_state[0],indices), tf.gather(cand_dec_state[1],indices))
			dec_index = self.beam_gen[:,-1]

		self.i += 1

		return dec_index, dec_state, self.beam_gen, self.beam_score

	def get_result(self):
		return self.beam_gen, self.beam_score, 

	def get_debug_result(self):
		pass
