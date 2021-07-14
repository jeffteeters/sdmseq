import numpy as np
import argparse
# import shlex
import sys
from random import randint
import time
import hashlib
import json
from bitarray import bitarray
import gmpy2


class Env:
	# stores environment settings and data arrays

	# command line arguments, also used for parsing interactive update of parameters
	parms = [
		{ "name":"word_length", "kw":{"help":"Word length for address and memory", "type":int},
	 	  "flag":"w", "required_init":"i", "default":512 },
	 	{ "name":"num_rows", "kw":{"help":"Number rows in memory","type":int},
	 	  "flag":"r", "required_init":"i", "default":2048 },
		{ "name":"activation_count", "kw":{"help":"Number memory rows to activate for each address","type":int},
		  "flag":"a", "required_init":"m", "default":20},
		{ "name":"debug", "kw":{"help":"Debug mode","type":int, "choices":[0, 1]},
		   "flag":"d", "required_init":"", "default":0},
		{ "name":"format", "kw":{"help":"Format used to store items and hard addresses, choices: "
		   "int8, np.packbits, bitarray, gmpy2, gmpy2pure, colsum"},
		  "flag":"f", "required_init":"i", "default":"int8", "choices":["int8", "np.packbits", "bitarray", "gmpy2",
		  "gmpy2pure"]},
		{ "name":"num_items", "kw":{"help":"Number items in item memory","type":int},
		  "flag":"i", "required_init":"m", "default":10},
		{ "name":"num_reps", "kw":{"help":"Number repepitions though itmes to match hard locations","type":int},
		  "flag":"t", "required_init":"m", "default":50},
		{ "name":"seed", "kw":{"help":"Random number seed","type":int},
		  "flag":"s", "required_init":"i", "default":2021},
		  ]

	def __init__(self):
		self.parse_command_arguments()
		self.display_settings()
		self.initialize()

	def parse_command_arguments(self):
		parser = argparse.ArgumentParser(description='Test formats for storing hard locations for a SDM.',
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		# also make parser for interactive updating parameters (does not include defaults)
		iparse = argparse.ArgumentParser(description='Update sdm parameters.') # exit_on_error=False)
		for p in self.parms:
			parser.add_argument("-"+p["flag"], "--"+p["name"], **p["kw"], default=p["default"])
			iparse.add_argument("-"+p["flag"], "--"+p["name"], **p["kw"])  # default not used for interactive update
		self.iparse = iparse # save for later parsing interactive input
		args = parser.parse_args()
		self.pvals = {p["name"]: getattr(args, p["name"]) for p in self.parms}

	def initialize(self):
		# initialize sdm, char_map and merge
		print("Initializing.")
		self.sdmt = SDMt(self)

	def display_settings(self):
		print("Current settings:")
		for p in self.parms:
			print(" %s: %s" % (p["name"], self.pvals[p["name"]]))

	def update_settings(self, line):
		instructions = ("Update settings using 'u' followed by KEY VALUE pair(s), where keys are:\n" +
			'; '.join(["-"+p["flag"] + " --"+p["name"] for p in self.parms]))
		#	" -w --world_length; -r --num_rows; -a --activation_count; -cmf --char_match_fraction; -s --string_to_store"
		if len(line) < 4:
			self.display_settings()
			print(instructions)
			return
		try:
			args = self.iparse.parse_args(shlex.split(line))
		except Exception as e:
		# except argparse.ArgumentError:
			print('Invalid entry, try again:\n%s' % s)
			return
		updated = []
		for p in self.parms:
			name = p["name"]
			val = getattr(args, name)
			if val is not None:
				if self.pvals[name] == val:
					print("%s unchanged (is already %s)" % (name, val))
				else:
					self.pvals[name] = val
					updated.append("%s=%s" % (name, val))
					if p["required_init"]:
						self.required_init += p["required_init"]
		if updated:
			print("Updated: %s" % ", ".join(updated))
			self.display_settings()
			print("required_init=%s" % self.required_init)
		else:
			print("Nothing updated")

class SDMt:
	# for testing SDM storage and recall
	def __init__(self, env):
		self.env = env
		np.random.seed(self.env.pvals["seed"])
		format = self.env.pvals["format"]
		self.hard_locations = initialize_binary_matrix(self.env.pvals["num_rows"],
			self.env.pvals["word_length"], format, debug=self.env.pvals["debug"])
		self.item_memory = initialize_binary_matrix(self.env.pvals["num_items"], 
			self.env.pvals["word_length"], format, items=True, debug=self.env.pvals["debug"])

	def run_test(self):
		result = ""
		format = self.env.pvals["format"]
		word_length = self.env.pvals["word_length"]
		num_rows = self.env.pvals["num_rows"]	
		start = time.process_time()
		for i in range(self.env.pvals["num_reps"]):
			for j in range(self.env.pvals["num_items"]):
				b = self.item_memory[j]
				top_matches = find_matches(self.hard_locations, b, self.env.pvals["activation_count"],
					format=format, word_length=word_length, num_rows=num_rows, debug=self.env.pvals["debug"])
				data_md5 = hashlib.md5(json.dumps(top_matches, sort_keys=True).encode('utf-8')).hexdigest()
				result += data_md5
		duration = time.process_time() - start
		data_md5 = hashlib.md5(result.encode('utf-8')).hexdigest()
		print("duration is %s s, data_md5: %s" % (duration, data_md5))

def as_bit_str(ba):
	# convert binary array to string of bits
	bstr = "";
	for b in ba:
		bstr += "%s" % b
	return bstr


def initialize_binary_matrix(nrows, ncols, format, items=False, debug=False):
	# create binary matrix with each row having binary random number
	# format one of: "int8", "np.packbits", "bitarray", "gimp2"
	# items is true if items, false if hard addresses
	bm = np.random.randint(2, size=(nrows, ncols), dtype=np.int8)
	byteorder = "big" # sys.byteorder
	if debug:
		print ("items=%s, bm=\n%s" % (items, bm))
	if format == "int8":
		pass
	elif format == "np.packbits":
		bm = np.packbits(bm, axis=-1)
	elif format == "bitarray":
		bm = [ bitarray(list(x.astype(bool))) for x in bm]
	elif format == "gmpy2":
		# convert numpy binary array to python integers
		bmp = np.packbits(bm, axis=-1)
		bm = []
		for b in bmp:
			intval = int.from_bytes(b.tobytes(), byteorder)
			if debug:
				print("bm intval=%s" % bin(intval))
			bm.append(intval)
	elif format == "gmpy2pure":
		bmp = np.packbits(bm, axis=-1)
		bm = []
		for b in bmp:
			intval = int.from_bytes(b.tobytes(), byteorder)
			impz = gmpy2.mpz(intval)
			bm.append(impz)
			if debug:
				print("bm intval=%s, impz=%s" % (bin(intval), impz.digits(2)))
	elif format == "colsum":
		# colsum uses gmpy2 mpz for items and numpy packbits (int8) for addresses
		# addresses are stored as in columns to allow finding hamming distance to each bit in item
		if not items:
			# storing address, transpose; store as np.packbits
			assert bm.shape[1] %8 == 0, "num address rows (hard locations) must be multiple of 8, is: %s" % (
				bm.shape[1])
			bm = np.transpose(bm)
			bm = np.packbits(bm, axis=-1)
		else:
			# storing items, store as mpz ints
			bmp = np.packbits(bm, axis=-1)
			bm = []
			for b in bmp:
				intval = int.from_bytes(b.tobytes(), byteorder)
				intval = gmpy2.mpz(intval)
				bm.append(intval)
				if debug:
					print("bm intval=%s" % intval.digits(2))
	else:
		sys.exit("unknown format: %s" % format)

		#b.dot(1 << np.arange(b.size)[::-1])

		# 4294967295
	# if not packed:
	# 	bm = np.random.randint(2, size=(nrows, ncols), dtype=np.int8)
	# else:
	# 	bm = np.random.randint(256, size=(nrows, (ncols / 8)), dtype=np.uint8)
	return bm

# from: https://stackoverflow.com/questions/40875282/fastest-way-to-get-hamming-distance-for-integer-array
# _nbits = np.array(
#       [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
#        4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
#        4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2,
#        3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
#        4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4,
#        5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
#        3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2,
#        3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
#        4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
#        6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5,
#        5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6,
#        7, 7, 8], dtype=np.uint8)

def find_matches(m, b, nret, format=None, word_length=None, num_rows=None, index_only = False, match_bits=None,
		debug=False):
	# m is 2-d array of binary values, first dimension is value index, second is binary number
	# b is binary number to match
	# word_length is number of bits in b (needed if stored using gmpy2.mpz format)
	# num_rows is number of rows (addresses).  Should equal m.shape[0]
	# nret is number of top matches to return
	# format specifies format of m and b
	# returns sorted tuple (i, c) where i is index and c is number of non-matching bits (0 is perfect match)
	# if index_only is True, only return the indices, not the c
	# if match_bits is not None, it is the number of bits to match (subset), otherwise full length is used
	matches = []
	if format == "int8":
		for i in range(m.shape[0]):
			ndiff = np.count_nonzero(m[i][0:match_bits]!=b[0:match_bits])
			matches.append( (i, ndiff) )
	elif format == "np.packbits":
		r = (1 << np.arange(8))[:,None]
		for i in range(len(m)):
			ndiff = np.count_nonzero((np.bitwise_xor(m[i],b) & r) != 0) # 10 sec
			# ndiff = int(np.unpackbits(np.bitwise_xor(m[i],b)).sum())  # 11 sec
			matches.append( (i, ndiff) )	
	elif format == "bitarray":
		for i in range(len(m)):
			ndiff =(m[i]^b).count()
			matches.append( (i, ndiff) )
	elif format == "gmpy2":
		# matches = np.zeros((num_rows,), dtype=np.int32)
		for i in range(len(m)):
			ndiff = gmpy2.popcount(m[i]^b)
			matches.append( (i, ndiff) )
			# matches[i] = ndiff
		# matches = list( zip (range(num_rows), matches.tolist()))
		# matches = [(i, int(matches[i])) for i in range(num_rows)]
			# matches.append( (i, ndiff) )
	elif format == "gmpy2pure":
		for i in range(len(m)):
			ndiff = gmpy2.hamdist(m[i],b)
			matches.append( (i, ndiff) )
	elif format == "colsum":
		act_num_rows, act_num_cols = m.shape
		assert act_num_rows == word_length
		assert act_num_cols * 8 ==  num_rows, "act_num_cols=%s, should be num_rows (%s)/8" % (act_num_cols, num_rows)
		if debug:
			print("m shape= (%s, %s) m is:" % m.shape)
			for ri in range(act_num_rows):
				print("%s" % [bin(m[ri, ci]) for ci in range(act_num_cols)])
		# compute xor on each column in hard location
		hdist = np.zeros(num_rows, dtype=np.uint16)
		if debug:
			print("b=%s" % b.digits(2))
			print("bits found:")
		for ri in range(word_length-1, -1, -1):  # ci - column index
			b1_set = b.bit_test(ri)
			hdi = 0
			bits_found = []
			for ci in range(act_num_cols-1, -1, -1):
				byte = m[word_length -1 - ri, ci]
				if debug:
					print("byte=%s" % bin(byte))
				mask = 128
				for i in range(8):
					b2_set = (byte & mask) != 0
					b1 = 1 if b1_set else 0
					b2 = 1 if b2_set else 0
					bits_found.append([b1, b2])
					# if b1_set ^ b2_set:
					if b1 != b2:
						hdist[hdi] += 1
					hdi += 1
					mask = mask >> 1
					# if debug:
					#	print("mask=%s" % mask)
			if debug:
				print("%s" % bits_found)
		matches = list( zip (range(num_rows), hdist.tolist()))
		# ndiff = [(i, hdist[i]) for i in range(hdist)]

		# 	assert len(list(m[i].iter_bits())) == num_addresses, "lengts didn't match, i=%s, len=%s" % (
		# 		i, len(list(m[i].iter_bits())))
		# for ib in b.iter_bits():
		# 	mcol = np.array(list(m[idx].iter_bits())) ^ ib
		# 	hdist += mcol
		# 	idx += 1


	if debug:
		print("matches =\n%s" % matches)
	# find top nret matches
	matches.sort(key = lambda y: (y[1], y[0]))
	top_matches = matches[0:nret]
	if index_only:
		top_matches = [x[0] for x in top_matches]
	return top_matches

	# if not packed:
	# 	assert len(m.shape) == 2, "array to match must be 2-d"
	# assert m.shape[1] == len(b), "array element size does not match size of match binary value"
	# if match_bits is None:
	# 	assert len(b) == m.shape[1], "match_bits is None but len(b) (%s) not equal to m.shape[1] (%s)" % (
	# 		len(b), m.shape[1])
	# 	match_bits = len(b)
	# assert match_bits <= len(b), "match_bits for find_matches too long (%s), must be less than (%s)" % (match_bits, len(b))
	# matches = []
	# if not packed:
	# 	for i in range(m.shape[0]):
	# 		ndiff = np.count_nonzero(m[i][0:match_bits]!=b[0:match_bits])
	# 		matches.append( (i, ndiff) )
	# else:
	# 	for i in range(len(m)):
	# 		ndiff =(m[i]^b).count()
	# 		matches.append( (i, ndiff) )

		# r = (1 << np.arange(8))[:,None]
		# for i in range(m.shape[0]):
		# 	# ndiff = np.count_nonzero((np.bitwise_xor(m[i],b) & r) != 0) # 10 sec
		# 	# ndiff = int(np.unpackbits(np.bitwise_xor(m[i],b)).sum())  # 11 sec
		# 	c = np.bitwise_xor(m[i],b)
		# 	ndiff = 23 # int(_nbits[c].sum())
		# 	matches.append( (i, ndiff) )
	# matches.sort(key=itemgetter(1))


def main():
	env = Env()
	env.sdmt.run_test()

main()