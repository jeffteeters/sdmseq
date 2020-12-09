# SDM model test simulation
import numpy as np
from operator import itemgetter
import pprint
pp = pprint.PrettyPrinter(indent=4)

class Sequences:
	# sequences learned
	def __init__(self, seq = ["happy_day", "evans_hall", "campanile", "sutardja_dai_hall", "oppenheimer"]):
		self.seq = seq


# def random_b(word_length):
# 	# return binary vector with equal number of 1's and 0's
# 	assert word_length % 2 == 0, "word_length must be even"
# 	hw = word_length / 2
# 	arr = np.array([1] * hw + [0] * hw, dtype=np.int8)
# 	np.random.shuffle(arr)
# 	return arr

def find_matches(m, b, nret, index_only = False):
	# m is 2-d array of binary values, first dimension if value index, second is binary number
	# b is binary number to match
	# nret is number of top matches to return
	# returns sorted tuple (i, c) where i is index and c is number of non-matching bits (0 is perfect match)
	# if index_only is True, only return the indices, not the c
	assert len(m.shape) == 2, "array to match must be 2-d"
	assert m.shape[1] == len(b), "array element size does not match size of match binary value"
	matches = []
	for i in range(m.shape[0]):
		ndiff = np.count_nonzero(m[i]!=b)
		matches.append( (i, ndiff) )
	matches.sort(key=itemgetter(1))
	top_matches = matches[0:nret]
	if index_only:
		top_matches = [x[0] for x in top_matches]
	return top_matches

def initialize_binary_matrix(nrows, ncols):
	# create binary matrix with each row having binary random number
	assert ncols % 2 == 0, "ncols must be even"
	bm = np.random.randint(2, size=(nrows, ncols), dtype=np.int8)
	# hw = ncols / 2
	# bm = np.zeros( (nrows, ncols), dtype=dtype=np.int8) # bm - binary_matrix
	# for i in range(nrows):
	# 	bm[i][0:hw] = 1				# set half of row to 1, then shuffle
	# 	np.random.shuffle(bm[i])
	return bm

def merge(b1, b2):
	# merge binary values b1, b2 by taking every other value of each and concationating
	b3 = np.concatenate((b1[0::2],b2[0::2]))
	return b3

class Char_map:
	# maps each character in a sequence to a random binary word
	def __init__(self, seq, word_length = 128):
		# seq - a Sequence object
		# word_length - number of bits in random binary word
		assert word_length % 2 == 0, "word_length must be even"
		self.word_length = word_length
		self.chars = ''.join(sorted(list(set(list(''.join(seq))))))  # sorted string of characters appearing in seq
		self.chars += "#"  # stop char - used to indicate end of sequence
		self.binary_vals = initialize_binary_matrix(len(self.chars), word_length)
	
	def char2bin(self, char):
		# return binary word associated with char
		index = self.chars.find(char)
		assert index != -1, "character '%s' not in char_map (%s)" % (char, self.chars)
		return self.binary_vals[index]

	def bin2char(self, b, nret = 3):
		# returns array of top matching characters and the hamming distance (# of bits not matching)
		# b is a binary array, same length as word_length
		# nret is the number of matches to return
		assert self.word_length == len(b)
		top_matches = find_matches(self.binary_vals, b, nret)
		top_matches = [ (self.chars[x[0]], x[1]) for x in top_matches ]
		return top_matches


class Sdm:
	# implements a sparse distributed memory
	def __init__(self, address_length=128, word_length=128, num_rows=512, nact=5):
		# nact - number of active addresses used (top matches) for reading or writing
		self.address_length = address_length
		self.word_length = word_length
		self.num_rows = num_rows
		self.nact = nact
		self.data_array = np.zeros((num_rows, word_length), dtype=np.int8)
		self.addresses = initialize_binary_matrix(num_rows, word_length)


	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		d = data.copy()
		d[d==0] = -1  # replace zeros in data with -1
		for i in top_matches:
			self.data_array[i] += d

	def read(self, address):
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		i = top_matches[0]
		sum = self.data_array[i].copy()
		for i in top_matches[1:]:
			sum += self.data_array[i]
		sum[sum<1] = 0   # replace values less than 1 with zero
		sum[sum>0] = 1   # replace values greater than 0 with 1
		return sum

def recall(prefix, sdm, cmap):
	# recall sequence starting with prefix
	# build address to start searching
	threshold = sdm.word_length / 4
	b0 = cmap.char2bin(prefix[0])  # binary word associated with first character
	address = merge(b0, b0)
	for char in prefix[1:]:
		b = cmap.char2bin(char)
		address = merge(b, address)
	found = [ prefix, ]
	word2 = prefix
	# now read sequence using prefix address
	while True:
		b = sdm.read(address)
		top_matches = cmap.bin2char(b)
		found.append(top_matches)
		found_char = top_matches[0][0]
		if found_char == "#":
			# found stop char
			break
		# perform "cleanup" by getting b corresponding to top match 
		if top_matches[0][1] > 0:
			# was not an exact match.  Check to see if hamming distance larger than threshold
			if top_matches[0][1] > threshold:
				break
			# Cleanup by getting b corresponding to top match
			b = cmap.char2bin(found_char)
		address = merge(b, address)
		word2 += found_char
	return [found, word2]

def main():
	word_length = 256
	num_rows = 512
	seq = ["happy day", "evans hall", "campanile", "sutardja dai hall", "oppenheimer", "distributed memory",
		"abcdefghijklmnopqrstuvwxyz"]
	seq = Sequences(seq)
	cmap = Char_map(seq.seq, word_length=word_length)
	sdm = Sdm(address_length=word_length, word_length=word_length, num_rows=num_rows)
	# store all sequences
	for word in seq.seq:
		print("storing '%s'" % word)
		b0 = cmap.char2bin(word[0])  # binary word associated with first character
		address = merge(b0, b0)
		for char in word[1:]+"#":  # add stop character at end
			b = cmap.char2bin(char)
			sdm.store(address, b)   # store code for char using address prev_bin
			address = merge(b, address)

	# retrieve sequences, starting with first character
	error_count = 0
	for word in seq.seq:
		print ("\nRecall '%s'" % word)
		found, word2 = recall(word[0], sdm, cmap)
		if word != word2:
			error_count += 1
			msg = "ERROR"
		else:
		    msg = "Match"
		pp.pprint(found)
		print ("Recall '%s'" % word)
		print("found: '%s' - %s" % (word2, msg))
	print("%s words, %s errors" % (len(seq.seq), error_count))

main()



		# found = [ word[0], ]
		# word2 = word[0]
		# b0 = cmap.char2bin(word[0])  # binary word associated with first character
		# address = merge(b0, b0)
		# for char in word[1:]:
		# 	b = sdm.read(address)
		# 	top_matches = cmap.bin2char(b)
		# 	found.append(top_matches)
		# 	found_char = top_matches[0][0]
		# 	word2 += found_char
		# 	# perform "cleanup" by getting b corresponding to top match 
		# 	if top_matches[0][1] > 0:
		# 		# was not an exact match.  Cleanup by getting b corresponding to top match
		# 		b = cmap.char2bin(top_matches[0][0])
		# 	address = merge(b, address)

