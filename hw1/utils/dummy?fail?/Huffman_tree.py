import heapq
from collections import defaultdict
import numpy as np

class Huffman_tree(object):
	def __init__(self, idx_count):
		def encode(frequency):
			print("Encoding by Huffman tree ...")
			heap = [[weight, [symbol, '']] for symbol, weight in frequency.items()]
			heapq.heapify(heap)
			while len(heap) > 1:
				lo = heapq.heappop(heap)
				hi = heapq.heappop(heap)
				for pair in lo[1:]:
					pair[1] = '0' + pair[1]
				for pair in hi[1:]:
					pair[1] = '1' + pair[1]
				heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
			return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

		self.idx2code = {}
		self.idx2nodes = {}
		self.max_code_len = None

		temp = encode(idx_count)
		temp_key = [t[0] for t in temp]
		temp_code = [t[1] for t in temp]
		self.idx2code = dict(zip(temp_key,temp_code))

		self.max_code_len = len(self.idx2code[len(self.idx2code)-1])

		pathes = []
		for key in self.idx2code:
			code = self.idx2code[key]
			pathes_this_code = [code[:i+1] for i in range(len(code[:-1]))]
			pathes +=  pathes_this_code

		pathes = list(set(pathes))
		pathes.sort()
		pathes.sort(key=len)
		path2node = dict(zip(pathes,range(1,len(pathes)+1)))

		for idx in self.idx2code:
			code = self.idx2code[idx] 
			pathes = [code[:i+1] for i in range(len(code[:-1]))]
			nodes = np.array([0] + [path2node[p] for p in pathes], dtype=np.int32)
			self.idx2nodes.update({idx:nodes})

	def process_batch_time_label_for_BiRNN(self,labels_padded):
		'''Return labels_direction for tf feed dict'''
		batchsize = len(labels_padded)
		maxtimestep = len(labels_padded[0])

		index_of_interest = np.empty((batchsize,maxtimestep-2,self.max_code_len,2),dtype=np.int32)
		node_direction_with_padding_by_two = np.empty((batchsize,maxtimestep-2,self.max_code_len),dtype=np.int32)

		for i in range(len(labels_padded)):
			sentence = labels_padded[i]
			for j in range(len(sentence[1:-1])): # For BiRNN, the first and the last word labels are not used for training
				wordid = sentence[j]
				code_len = len(self.idx2nodes[wordid])
				nodes_padded = np.hstack((self.idx2nodes[wordid],np.zeros(self.max_code_len - code_len))) #[0,2,5,...,59997,0,0,0]
				flatten_index = i*len(sentence[1:-1])+j
				indexed_nodes_padded = [[flatten_index, node] for node in nodes_padded] #[[3,0],[3,2],[3,5],...,[3,59997],[3,0],[3,0],[3,0]]
				index_of_interest[i,j] = indexed_nodes_padded[:]

				code2int_padded_by_two = [int(c) for c in self.idx2code[wordid]] + [2]*(self.max_code_len - code_len)
				node_direction_with_padding_by_two[i,j] = code2int_padded_by_two[:]
 
		#labels_nodes = [[self.idx2nodes(idx) for idx in b] for b in labels_padded] # (batch_size, max_time_step, max_code_len)
		#labels_nodes_indexed = [[[[(n-1)+m*(len(labels_nodes[m])-2),node] for node in labels_nodes[m][n]] for n in range(1,len(labels_nodes[m])-1)] for m in range(len(labels_nodes))] # (batch_size , max_time_step-2, max_code_len, 2)
		#labels_direction = [[self.idx2code_int_padded[idx] for idx in b] for b in labels_padded] # (batch_size, max_time_step, max_code_len)

		return index_of_interest, node_direction_with_padding_by_two
