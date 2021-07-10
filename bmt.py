import numpy as np
import argparse
# import shlex
# import sys
from random import randint
import time
import hashlib
import json


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
		{ "name":"stored_format", "kw":{"help":"Format used to store hard addresses, must be 'int8' or 'packed'"},
		  "flag":"f", "required_init":"i", "default":"int8", "choices":["int8", "packed"]},
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
		self.packed = self.env.pvals["stored_format"] == "packed"
		np.random.seed(self.env.pvals["seed"])
		self.hard_locations = initialize_binary_matrix(self.env.pvals["num_rows"], self.env.pvals["word_length"], self.packed)
		self.item_memory = initialize_binary_matrix(self.env.pvals["num_items"], self.env.pvals["word_length"], self.packed)

	def run_test(self):
		result = ""
		start = time.process_time()
		for i in range(self.env.pvals["num_reps"]):
			for j in range(self.env.pvals["num_items"]):
				b = self.item_memory[j]
				top_matches = find_matches(self.hard_locations, b, self.env.pvals["activation_count"], self.packed)
				data_md5 = hashlib.md5(json.dumps(top_matches, sort_keys=True).encode('utf-8')).hexdigest()
				result += data_md5
		duration = time.process_time() - start
		data_md5 = hashlib.md5(result.encode('utf-8')).hexdigest()
		print("duration is %s s, data_md5: %s" % (duration, data_md5))

def initialize_binary_matrix(nrows, ncols, packed):
	# create binary matrix with each row having binary random number
	# if packed, store as binary, otherwise as bytes
	assert ncols % 8 == 0, "ncols must be multiple of 8"
	bm = np.random.randint(2, size=(nrows, ncols), dtype=np.int8)
	if packed:
		# pack so 
		bm = np.packbits(bm, axis=-1)
	# if not packed:
	# 	bm = np.random.randint(2, size=(nrows, ncols), dtype=np.int8)
	# else:
	# 	bm = np.random.randint(256, size=(nrows, (ncols / 8)), dtype=np.uint8)
	return bm

# from: https://stackoverflow.com/questions/40875282/fastest-way-to-get-hamming-distance-for-integer-array
_nbits = np.array(
      [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3,
       4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
       4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2,
       3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
       4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4,
       5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
       3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2,
       3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
       4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
       6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5,
       5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6,
       7, 7, 8], dtype=np.uint8)

def find_matches(m, b, nret, packed=False, index_only = False, match_bits=None):
	# m is 2-d array of binary values, first dimension is value index, second is binary number
	# b is binary number to match
	# nret is number of top matches to return
	# packed is True if numbers are packed in binary, otherwise as bytes (int8)
	# returns sorted tuple (i, c) where i is index and c is number of non-matching bits (0 is perfect match)
	# if index_only is True, only return the indices, not the c
	# if match_bits is not None, it is the number of bits to match (subset), otherwise full length is used
	assert len(m.shape) == 2, "array to match must be 2-d"
	# assert m.shape[1] == len(b), "array element size does not match size of match binary value"
	if match_bits is None:
		assert len(b) == m.shape[1], "match_bits is None but len(b) (%s) not equal to m.shape[1] (%s)" % (
			len(b), m.shape[1])
		match_bits = len(b)
	assert match_bits <= len(b), "match_bits for find_matches too long (%s), must be less than (%s)" % (match_bits, len(b))
	matches = []
	if not packed:
		for i in range(m.shape[0]):
			ndiff = np.count_nonzero(m[i][0:match_bits]!=b[0:match_bits])
			matches.append( (i, ndiff) )
	else:
		r = (1 << np.arange(8))[:,None]
		for i in range(m.shape[0]):
			# ndiff = np.count_nonzero((np.bitwise_xor(m[i],b) & r) != 0) # 10 sec
			# ndiff = int(np.unpackbits(np.bitwise_xor(m[i],b)).sum())  # 11 sec
			c = np.bitwise_xor(m[i],b)
			ndiff = 23 # int(_nbits[c].sum())
			matches.append( (i, ndiff) )
	# matches.sort(key=itemgetter(1))
	matches.sort(key = lambda y: (y[1], y[0]))
	top_matches = matches[0:nret]
	if index_only:
		top_matches = [x[0] for x in top_matches]
	return top_matches

def main():
	env = Env()
	env.sdmt.run_test()

main()