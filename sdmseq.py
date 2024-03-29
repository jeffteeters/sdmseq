# SDM model test simulation
import numpy as np
from operator import itemgetter
import pprint
pp = pprint.PrettyPrinter(indent=4)
import argparse
import shlex
import readline
readline.parse_and_bind('tab: complete')
import sys
from random import randint


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
	 	{ "name":"char_match_fraction", "kw": {"help":"Fraction of word_length to form hamming distance threshold for"
			" matching character to item memory","type":float},"flag":"cmf", "required_init":"", "default":0.3},
		{ "name":"merge_algorithm", "kw":{"help":"Algorithm used combine item and history when forming new address. "
		    "wx - Weighted and/or XOR, wx2 - save xor with data, fl - First/last bits, hh - concate every other bit",
		    "choices":["wx", "wx2", "fl", "hh", "hh2"]},
	 	  "flag":"ma", "required_init":"m", "default":"wx2"},
		# { "name":"permute", "kw":{"help":"Permute values when storing","type":int, "choices":[0, 1]},
		# 	  "flag":"p", "required_init":True, "default":0},
		{ "name":"start_bit", "kw":{"help":"Starting bit for debugging","type":int, "choices":[0, 1]},
		  "flag":"b", "required_init":"m", "default":0},
	 	{ "name":"history_fraction", "kw":{"help":"Fraction of history to store when forming new address "
	 	    "(Both wx and fl algorighms)", "type":float},
	 	    "flag":"hf", "required_init":"m", "default":0.5},
	 	{ "name":"xor_fraction", "kw":{"help":"Fraction of bits used for xor component in wx algorithm","type":float},
	 	  "flag":"xf", "required_init":"m", "default":0.25},
	 	{ "name":"first_bin_fraction", "kw":{"help":"First bin fraction, fraction of bits of item used for first bin "
	 	  "in wx algorithm OR an integer > 1 giving number of equal history bins (wx2 algorithm)",
	 	  "type":float}, "flag":"fbf", "required_init":"m", "default":8},
	 	{ "name":"converge_count", "kw":{"help":"Converge count for wx2; format: <int1>,<int2>; <int1> number of reads to converge "
	 	  "for each seed addresss, <int2> number of seeds to try", "type":str}, "flag":"cvc",
	 	  "required_init":"", "default":"30,30"},
	 	{ "name":"seeded_shuffle", "kw":{"help":"Shuffle history part using seed (wx2 algorithm), 0-no, 2-yes (new method)",
	 	  "type":int, "choices":[0, 2]},"flag":"ss", "required_init":"m", "default":2},
	 	{ "name":"recall_fix", "kw":{"help":"Fix errors when recalling sequence (wx2 algorithm), 1-yes, 0-no",
	 	  "type":int, "choices":[0, 1]},"flag":"rf", "required_init":"", "default":0},
	 	{ "name":"min_key_length", "kw":{"help":"minimum length of key string recalling (wx2 algorithm), must be >1",
	 	  "type":int},"flag":"mkl", "required_init":"m", "default":4},
	 	{ "name":"num_target_bins", "kw":{"help":"Num bins in value to use for next and prev chars (wx2 algorithm). "
	 	     "Range 0 to min_key_length.  Zero means min_key_length (all).  Normally, should be all, but can be made "
	 	     "smaller to cause errors to test recall_fix.", "type":int},"flag":"ntb", "required_init":"m", "default":0},
	 	{ "name":"allow_reverse_recall", "kw":{"help":"Allow reverse recalling sequence (wx2 algorithm), 1-yes, 0-no",
	 	  "type":int, "choices":[0, 1]},"flag":"rev", "required_init":"m", "default":0},
	 	{ "name":"debug", "kw":{"help":"Debug mode","type":int, "choices":[0, 1]},
		   "flag":"d", "required_init":"", "default":0},
		{ "name":"string_to_store", "kw":{"help":"String to store","type":str,"nargs":'*'}, "required_init":"",
		  "flag":"s", "default":
		  # '"cory qqqq eecs" "evans qqqq math" '
		  # '"abcdefghijklmnopqrstuvwxyz" '
		  '"tom jones teaches biology" "sue davis teaches economics"'
		  # '"stanley hall 20*rrrrrrrrrrrrrrrrrrrr biology" '
		  # '"bechtel hall 20*rrrrrrrrrrrrrrrrrrrr engineering"'
		  # '"happy day" "evans hall" "campanile" "sutardja dai hall" "oppenheimer"'
		  #' "distributed memory" "abcdefghijklmnopqrstuvwxyz"'
		  # ' "Evans rrrrr math" "Corey rrrrr eecs"'
		  }]

	def __init__(self):
		self.parse_command_arguments()
		self.initialize()
		self.display_settings()

	def parse_command_arguments(self):
		parser = argparse.ArgumentParser(description='Store sequences using a sparse distributed memory.',
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		# also make parser for interactive updating parameters (does not include defaults)
		iparse = argparse.ArgumentParser(description='Update sdm parameters.') # exit_on_error=False)
		for p in self.parms:
			parser.add_argument("-"+p["flag"], "--"+p["name"], **p["kw"], default=p["default"])
			iparse.add_argument("-"+p["flag"], "--"+p["name"], **p["kw"])  # default not used for interactive update
		self.iparse = iparse # save for later parsing interactive input
		args = parser.parse_args()
		self.pvals = {p["name"]: getattr(args, p["name"]) for p in self.parms}
		self.pvals["string_to_store"] = shlex.split(self.pvals["string_to_store"])

	def initialize(self):
		# initialize sdm, char_map and merge
		word_length = self.pvals["word_length"]
		debug = self.pvals["debug"]
		print("Initializing.")
		self.cmap = Char_map(self.pvals["string_to_store"], word_length=word_length, debug=debug)
		self.sdm = Sdm(address_length=word_length, word_length=word_length, num_rows=self.pvals["num_rows"],
			nact = self.pvals["activation_count"], debug=self.pvals["debug"])
		self.initialize_merge_algorithm()
		# self.merge = Merge(self)
		self.saved_strings = []
		self.required_init = ""  # contains chars: 'i' - initialize structures, 'm' - clear memory, '' - none-needed

	def initialize_merge_algorithm(self):
		self.ma = getattr(sys.modules[__name__], "Ma_" + self.pvals["merge_algorithm"])(self)

	def record_saved_string(self, string):
		self.saved_strings.append(string)

	def ensure_initialized(self):
		# make sure initialized or memory cleared after parameter changes
		if "i" in self.required_init:
			self.initialize()
		elif "m" in self.required_init:
			self.clear()
		else:
			assert self.required_init == "", "invalid char in required_init: %s" % self.required_init

	def clear(self):
		if 'i' in self.required_init:
			# must initialize rather than clear
			print("Initialization required before clearing.")
			self.initialize()
		else:
			print("Clearing memory.")
			self.sdm.clear()
			self.initialize_merge_algorithm()
			self.saved_strings = []
			self.required_init = ""

	def display_settings(self):
		print("Current settings:")
		for p in self.parms:
			print(" %s: %s" % (p["name"], self.pvals[p["name"]]))
		print("Saved strings=%s" % self.saved_strings)

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


def do_interactive_commands(env):
	instructions = ("Enter command, control-d to quit\n"
		" s <string> - store new string(s) OR param strings (if none specified)\n"
		" r <prefix> - recall starting with prefix\n"
		" r - recall param strings\n"
		" v <prefix> - recall in reverse order\n"
		" u - update parameters\n"
		" c - clear memory\n"
		" t - test merge convergence\n"
		" h - show hits when storing\n"
		" i - initalize everything")
	print(instructions)
	while True:
		try:
			line=input("> ")
		except EOFError:
			break;
		if len(line) == 0 or line[0] not in "sruictvh":
			print(instructions)
			continue
		cmd = line[0]
		arg = "" if len(line) == 1 else line[1:].strip()
		if cmd == "u":
			print("update parameter settings %s" % arg)
			env.update_settings(arg)
			continue
		if cmd == "i":
			env.initialize()
			continue
		if cmd == "c":
			env.clear()
			continue
		if cmd == "h":
			env.sdm.show_hits()
			continue
		# remaining commands may requre reinitializing or memory clearing if parameter settings have changed 
		env.ensure_initialized()
		if cmd == "r" and len(arg) > 0:
			print("recall prefix %s" % arg)
			recall_strings(env, shlex.split(arg), prefix_mode=True)
		elif cmd == "v" and len(arg) > 0:
			print("reverse recall prefix %s" % arg)
			recall_strings(env, shlex.split(arg), prefix_mode=True, reverse=True)
		elif cmd == "r":
			print("recall stored strings")
			recall_param_strings(env)
		elif cmd == "s" and len(arg) > 0:
			print("store new string %s" % arg)
			store_strings(env, shlex.split(arg))
		elif cmd == "s":
			print("store '-s' strings")
			store_param_strings(env)
		elif cmd == "t":
			print("Testing merge algorithm (%s) convergence" % env.pvals["merge_algorithm"])
			test_merge_convergence(env)
		else:
			sys.exit("Invalid command: %s" % cmd)
	print("\nDone")

# def random_b(word_length):
# 	# return binary vector with equal number of 1's and 0's
# 	assert word_length % 2 == 0, "word_length must be even"
# 	hw = word_length / 2
# 	arr = np.array([1] * hw + [0] * hw, dtype=np.int8)
# 	np.random.shuffle(arr)
# 	return arr

def find_matches(m, b, nret, index_only = False, match_bits=None):
	# m is 2-d array of binary values, first dimension is value index, second is binary number
	# b is binary number to match
	# nret is number of top matches to return
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
	for i in range(m.shape[0]):
		ndiff = np.count_nonzero(m[i][0:match_bits]!=b[0:match_bits])
		matches.append( (i, ndiff) )
	# matches.sort(key=itemgetter(1))
	matches.sort(key = lambda y: (y[1], y[0]))
	top_matches = matches[0:nret]
	if index_only:
		top_matches = [x[0] for x in top_matches]
	return top_matches

def initialize_binary_matrix(nrows, ncols, force_unique=False):
	# create binary matrix with each row having binary random number
	# if force_unique is True, fource first four bits be unique (used for debugging)
	assert ncols % 2 == 0, "ncols must be even"
	bm = np.random.randint(2, size=(nrows, ncols), dtype=np.int8)
	if force_unique:
		vals = np.arange(16)
		np.random.shuffle(vals)
		for i in range(len(vals)):
			val = vals[i]
			vala = np.unpackbits(np.uint8(val))
			bm[i][0:4] = vala[4:8]
	return bm

def make_permutation_map(length):
	# create permutation map for history vector.  This done because the numpy function I found
	# np.random.RandomState(seed=42).permutation(b) does not always include all the elements in
	# the permutation
	# the routine here works by having shifts go through order of indexes that are random
	shuffled_indices = np.arange(length, dtype=np.int32)
	np.random.shuffle(shuffled_indices)
	return make_index_map(shuffled_indices)

def make_index_map(shuffled_indices, inverse=False):
	# map each index to the next one in the shuffled list.  If inverse, map the other direction
	length = len(shuffled_indices)
	index_map = np.full(length, -1, dtype=np.int32)
	for i in range(length):
		from_index = shuffled_indices[i]
		to_index = shuffled_indices[(i+1) % length]
		if inverse:
			from_index, to_index = [to_index, from_index] # swap values
		assert index_map[to_index] == -1
		index_map[to_index] = from_index
	assert len(np.where(index_map==-1)[0]) == 0, "Did not fill all values in index_map"
	# map_index = [shuffled_indices[(i+1) % length] for i in range(length)]
	return index_map

def make_permutation_map_orig(length):
	# create permutation map for history vector.  This done because the numpy function I found
	# np.random.RandomState(seed=42).permutation(b) does not always include all the elements in
	# the permutation
	# NOTE: This routine is not correct, can create a non-cyclic permutation.  But it has allowed
	# distinguishing these two long strings:
	# 'tom smith is studying at the university of california at berkeley in the department of math'
	# 'sue jones is studying at the university of california at berkeley in the department of eecs'
	shuffled_indices = np.arange(length)
	np.random.shuffle(shuffled_indices)
	# map each index to the next one in the shuffled list
	map_index = [shuffled_indices[(i+1) % length] for i in range(length)]
	return map_index

def seeded_shuffle(b, seed, inverse=False):
	# shuffle elements in array b using array seed.  Both b and seed are arrays of binary values
	# if inverse is True, create shuffle which is inverse of shuffle when inverse=True
	blen = len(b)
	indices = list(range(blen))
	wl = blen.bit_length()
	shuffled_indices = np.full(blen, -1, dtype=np.int32)
	slen = len(seed)
	for i in range(blen):
		sidx = "".join([str(seed[(i*wl+j)%slen]) for j in range(wl)])
		idx = int(sidx, 2)  # convert binary string to int, e.g. "0110" ==> 6
		idx = idx % len(indices)  # put in range to indicies
		shuffled_indices[i] = indices.pop(idx)
	assert len(np.where(shuffled_indices==-1)[0]) == 0
	index_map = make_index_map(shuffled_indices, inverse)
	shuffled = b[index_map]
	return shuffled


class Merge_algorithm():
	# Abstract class for algorithms used for creating address and data to store and recall sequences

	def __init__(self,env):
		self.env = env
		name = type(self).__name__
		prefix = "Ma_"
		assert name.startswith(prefix)
		self.name = name[len(prefix):]
		self.pvals = {}
		self.initialize()

	def make_initial_address(self, bchar0):
		# make initial address for starting sequence.  wx2 overrides this
		return self.make_new_address(bchar0, bchar0)


	def make_initial_value(self, address, bchar1):
		# make value associated with address.  wx2 overides this
		return bchar1

	# def make_initial_address_old(self, char):
	# 	# returns (address, value) corresponding to starting character char
	# 	bchar = self.env.cmap.char2bin(char)
	# 	return (bchar, bchar)	


	# def add_old(self, address, value, char, cindex):
	# 	# add character char to sequence
	# 	# address and value is current address and value at that address before adding char to sequence
	# 	# cindex is the index of char in the sequence (zero based)
	# 	# Returns (new_address, new_value) - new address and the value to store at that address
	# 	bchar2 = self.env.cmap.char2bin(char)
	# 	bchar1 = self.get_bchar1(value)
	# 	new_address = self.make_new_address(address, bchar1)
	# 	new_value = self.make_new_value(address, new_address, bchar2, cindex)
	# 	return (new_address, new_value)

	def next(self, address, value):
		# get next address and character in sequence
		# given address and value at that address, return [new_address, char, top_matches] character in sequence
		bchar1_part = self.get_bchar1_part(value)
		top_matches = self.env.cmap.bin2char(bchar1_part, nret=6, match_bits=len(bchar1_part))
		char = top_matches[0][0]
		if char == "#" or top_matches[0][1] > int(self.env.pvals["char_match_fraction"] * len(bchar1_part)):
			# found stop char, or hamming distance to top match is over threshold
			new_address = None
		else:
			if top_matches[0][1] > 0 or len(bchar1_part) < self.env.pvals["word_length"]:
				# get cleaned up binary for bchar1
				bchar1_part = self.env.cmap.char2bin(char)
			new_address = self.make_new_address(address, bchar1_part)
		return [new_address, char, top_matches]

	def prev(self, address, value):
		# get previous address and character in sequence (reverse recall)
		# given address and value at that address, return [new_address, char] character in reverse sequence
		if self.name != "wx2":
			print("function 'prev' of merge_algorithm '%s' not implementd" % self.name)
			return [None, None, None]
		if not self.pvals["include_reverse_char"]:
			print("include_reverse_char not set; to set it make first_bin_fraction integer > 1 and 'enable_reverse' == 1 %s)" %
				self.env.pvals["first_bin_fraction"])
			return [None, None, None]
		bchar0_part = self.get_bchar0_part(address)
		top_matches = self.env.cmap.bin2char(bchar0_part, nret=6, match_bits=len(bchar0_part))
		char = top_matches[0][0]
		if top_matches[0][1] > int(self.env.pvals["char_match_fraction"] * len(bchar0_part)):
			# found stop char, or hamming distance to top match is over threshold
			new_address = None
		else:
			new_address = self.make_reverse_address(address, value, char, top_matches)
			if new_address is None:
				char = "[" + self.extract_sequence_prefix(address) + "]" # + char  # don't include char because it's in the prefix
		return [new_address, char, top_matches]

	# following default functions may be overridden in subclassses

	def initialize(self):
		print("Initialize merge_algorithm '%s'" % self.name)

	def get_bchar1_part(self, value):
		# return bits of bchar1 in value (but don't expand to full length if not already full length)
		return value

	def get_bchar1(self, value):
		# return full length
		return value

	def make_new_value(self, address, new_address, bchar2, cindex):
		return bchar2

	# following function must be provided in subclass to implement the merge_algorithm
	# def make_new_address(self, address, bchar1_part):
		# print("function 'make_new_address' of merge_algorithm '%s' not implementd" % self.name)
	# following functions may be provided to allow reverse sequence recall
	# def reverse_end_found(self, address, value, char):
	# def make_reverse_address(self, address, value, char, top_matches):



class Ma_wx(Merge_algorithm):

	def initialize(self):
		# wx Weighted and/or XOR algorithm: created address as two parts: weighted history followed by xor
		# number bits in xor part is xf*N (xf is xor_fraction)
		# weighted history part formed by selecting (1-hf) N bits from item and hf*N bits from history.
		# hf is fraction of history to store when forming new address
		# this function builds the maps selecting bits for the history (not the xor component)
		xf = self.env.pvals["xor_fraction"]
		word_length = self.env.pvals["word_length"]
		xr_len = int(word_length * xf)
		wh_len = word_length - xr_len
		char_match_threshold = word_length * self.env.pvals["char_match_fraction"]
		self.pvals = {"char_match_threshold": char_match_threshold}
		if self.name == "wx2" and (wh_len == 0 or xr_len == 0):
			print("** Error in wx2 settings, wh_len (%s) and xr_len (%s) must both be > 0" % (wh_len, xr_len))
		if wh_len > 0:
			# compute weighted history component
			# has two parts, new item bits, then history bits
			# this code has a bug, may not work unless history_fraction == 0.5
			fbf = self.env.pvals["first_bin_fraction"]  # fraction of wh_len used for first bin (bits from current item)
			include_reverse_char = self.name == "wx2" and fbf > 1.0 and self.env.pvals["allow_reverse_recall"]
			self.pvals.update( {"include_reverse_char": include_reverse_char} )
			if fbf > 1.0:
				# equal size bins
				num_bins = int(fbf)
				bits_per_bin = int(round(wh_len/num_bins))
				if num_bins * bits_per_bin != wh_len:
					# change size of xr_len and wh_len so all bins are the same size
					wh_len = num_bins * bits_per_bin
					xr_len = word_length - wh_len
				remaining_bits = wh_len % num_bins
				assert remaining_bits == 0
				# following more complex then needed since bins are all the same size
				bin_sizes = [ bits_per_bin + (1 if i < remaining_bits else 0) for i in range(num_bins)]
				bin0_len = bin_sizes[0]
			else:
				# bin sizes are specified by exponential decay (hf) of initial bin 
				hf = self.env.pvals["history_fraction"]
				bin_sizes = []
				bin0_len = int(fbf * wh_len)
				bits_left = wh_len - bin0_len
				bin_sizes.append(bin0_len)
				ibin = 0
				min_bin_size = 4
				while bits_left > min_bin_size:
					next_bin_size = round(bin_sizes[ibin] * hf)
					if next_bin_size  < min_bin_size:
						next_bin_size = min(min_bin_size, bits_left)
					bin_sizes.append(next_bin_size)
					bits_left -= next_bin_size
					ibin += 1
				if bits_left > 0:
					# distribute remaining bits in previous bins
					# from: https://stackoverflow.com/questions/21713631/distribute-items-in-buckets-equally-best-effort
					new_bits_per_bin = int(bits_left / (len(bin_sizes) - 1))
					remaining_bits = bits_left % (len(bin_sizes) - 1)
					for ibin in range(1, len(bin_sizes)):
						extra = 1 if ibin <= remaining_bits else 0
						bin_sizes[ibin] += new_bits_per_bin + extra
			# create indexing map to move bits each iteration using fancy indexing
			ind = []
			offset = 0
			for i in range(1,len(bin_sizes)):
				bin_size = bin_sizes[i]
				for j in range(bin_size):
					ind.append(j+offset)
				offset += bin_sizes[i-1]
			assert len(ind) + bin0_len + xr_len == word_length, ("wx hist init mismatch, ind=%s, bin0_len=%s, xr_len=%s" %
				(ind, bin0_len, xr_len))
			print("wx init:")
			print("item_len=%s" % bin0_len)
			print("bin_sizes=%s" % bin_sizes)
			print("hist_bits=%s" % ind)
			self.pvals.update( {"hist_bits": ind, "item_len":bin0_len, "bin_sizes":bin_sizes})
			if self.name == "wx2":
				mkl = self.env.pvals["min_key_length"]
				assert mkl > 1
				ntb = self.env.pvals["num_target_bins"]
				if ntb == 0:
					ntb = mkl  # defalut value (all bins)
				if ntb > mkl:
					ntb = mkl  # cannot be more than available bins
				targets_len = ntb * bin0_len
				if include_reverse_char:
					len_item_part = round(targets_len / 2.0)
				else:
					len_item_part = targets_len
				len_reverse_part = targets_len - len_item_part
				self.pvals.update( { "len_item_part":len_item_part, "len_reverse_part":len_reverse_part,
					"targets_len":targets_len })
				if include_reverse_char:
					# used for wx2 if allow reverse recall
					last_char_position = sum(bin_sizes[0:-1])  # position of reverse character
					empty_flag = self.env.cmap.char2bin("#")[0:len_reverse_part]  # bits for end of string character
					self.pvals.update( {"last_char_position":last_char_position, "empty_flag":empty_flag} )
		self.pvals.update( {"wh_len":wh_len, "xr_len":xr_len})

	# def add(self, address, value, char):
	# 	# add character char to sequence
	# 	# address and value is current address and value at that address before adding char to sequence
	# 	# Returns (new_address, new_value) - new address and the value to store at that address
	# 	# Fields in address are:
	# 	#  <history_part> <xor_part>
	# 	#  <history_part> of address is:
	# 	#     <item_t> <item_t-1> <item_t-2> ... <item_t-n+1>    (assuming n bins)
	# 	# Fields in value for wx2 class are:
	# 	#     <item_t+1> [<item_t-n>] <item_t-1> <item_t-2> ... <item_t-n+1>  <xor_part> (aligns with address)
	# 	#  OR, for wx class, simply, <item_t+1>  (no <xor part>)
	# 	# If <item_t-n> is present, size of <item_t+1> and <item_t-n> are reduced by half to allow alignment with address
	# 	# Generated new_address has fields:
	# 	#    <item_t+1> <item_t> <item_t-1> ... <item_t-n>  <new xor_part>
	# 	#       (It's made by shifting history_part in previous address right, inserting item from value)
	# 	# Generated new_value has fields:
	# 	#  <char> [<item_t-n+1>] <item_t> <item_t-1> ... <item_t-n> <new xor_part>
	# 	# where:
	# 	#  <char> is bits for input char (<item_t+2>)
	# 	#  <new xor_part> formed by shifting <xor_part> right, xor with <item_t+1> (item in input value)



	def make_new_address(self, address, bchar1):
		# form address as two components.  First (1-xf*N) bits are weighted history. Remaining bits (xf*N)are permuted XOR.
		if(self.pvals["wh_len"] > 0):
			# select bits specified by hist_bits and first part of item
			hist_part = np.concatenate((bchar1[0:self.pvals["item_len"]], address[self.pvals["hist_bits"]]))
			if self.env.pvals["first_bin_fraction"] > 1:
				# test to make sure fancy_indexing is working
				last_char_position = sum(self.pvals["bin_sizes"][0:-1])
				hist2_part = np.concatenate((bchar1[0:self.pvals["item_len"]],
					address[0:last_char_position]))
				if hamming(hist_part, hist2_part) != 0:
					print("difference found in hist_bits vs direct shift in wx2 make_new_address")
					import pdb; pdb.set_trace()
			if self.pvals["xr_len"] == 0:
				# no xor component
				assert len(hist_part) == self.env.pvals["word_length"], ("wx algorithm, no xor part, but len(hist_part) %s"
					" does not match word_length (%s)" % (len(hist_part), self.env.pvals["word_length"]))
				return hist_part
		# compute XOR part.  Is permute ( xor component from address) XOR leading bits from <item_t+1>.
		# if wh_len is zero then is pure XOR algorithm
		addr_input = address[self.pvals["wh_len"]:]
		item_input = bchar1[0:self.pvals["xr_len"]]
		assert len(addr_input) == len(item_input)
		xor_part = np.bitwise_xor(np.roll(addr_input, 1), item_input)
		new_address = np.concatenate((hist_part, xor_part)) if self.pvals["wh_len"] > 0 else xor_part
		assert len(new_address) == self.env.pvals["word_length"], "wx algorithm, len(new_address) (%s) != word_length (%s)" % (
			len(new_address), self.env.pvals["word_length"])
		return new_address


class Ma_wx2(Ma_wx):
	# similar to wx, except data value includes xor part

	def make_initial_address(self, bchar0):
		# make initial address for starting sequence
		# wx2 algorithm stores char shifted and xor'd with self to make xor part
		# stores prefix of xor part after all bins with valid content
		assert self.pvals["wh_len"] > 0 and self.pvals["xr_len"] > 0, "wx2 algorithm requires wh_len and xr_len > 0"
		assert self.env.pvals["first_bin_fraction"] > 1, "wx2 algorithm now requires equal bin sizes"
		xor_part = bchar0[0:self.pvals["xr_len"]]
		xor_part = np.bitwise_xor(np.roll(xor_part, 1), xor_part)
		hist_part = bchar0[0:self.pvals["wh_len"]]
		initial_address = np.concatenate((hist_part, xor_part))
		assert len(initial_address) == self.env.pvals["word_length"]
		return initial_address

	def make_initial_value(self, address, bchar2):
		# make value associated with initial address
		assert self.name == "wx2"
		if self.pvals["include_reverse_char"]:
			targets = np.concatenate((bchar2[0:self.pvals["len_item_part"]], self.pvals["empty_flag"]))
		else:
			targets = bchar2[0:self.pvals["len_item_part"]]
		value = np.concatenate((targets, address[self.pvals["targets_len"]:]))
		assert len(value) == self.env.pvals["word_length"], ("len(value)=%s, len(targets)==%s, len_item_part=%s,"
			"targets_len=%s" % (len(value), len(targets), self.pvals["len_item_part"],self.pvals["targets_len"]))
		return value

	# def make_new_address_shuffel_draft(self, address, bchar1):
	# 	# for wx2, insert item and shift history all one bin (bins are same size)
	# 	wh_len = self.pvals["wh_len"]
	# 	item_len = self.pvals["item_len"]
	# 	item_part = bchar1[0:item_len]
	# 	hist_part = address[0:wh_len-item_len]
	# 	new_hist_part = seeded_shuffle(np.concatenate((item_part, hist_part)), bchar1)
	# 	xor_part = address[wh_len:]
	# 	xor_part = np.bitwise_xor(np.roll(xor_part, 1), bchar1[0:self.pvals["xr_len"]])
	# 	new_address = np.concatenate((new_hist_part, xor_part))
	# 	return new_address

	def make_new_address(self, address, bchar1):
		# for wx2, insert item and shift history all one bin (bins are same size)
		wh_len = self.pvals["wh_len"]
		item_len = self.pvals["item_len"]
		item_part = bchar1[0:item_len]
		hist_part = address[0:wh_len-item_len]
		xor_part = address[wh_len:]
		xor_part = np.bitwise_xor(np.roll(xor_part, 1), bchar1[0:self.pvals["xr_len"]])
		new_address = np.concatenate((item_part, hist_part, xor_part))
		return new_address

	def get_bchar1_part(self, value):
		# get bits for <char_t+1> from value
		assert self.name == "wx2"
		bc1p_len = self.pvals["len_item_part"]
		return value[0:bc1p_len]

	def get_bchar1(self, value):
		bchar1_part = self.get_bchar1_part(value)
		match_bits = len(bchar1_part)
		bchar1 = self.env.cmap.part2full(bchar1_part, match_bits = match_bits)
		return bchar1

	def make_new_value(self, address, new_address, bchar2, cindex):
		# bchar2 is item to store in value (next character in sequence)
		# cindex is 0-based index of bchar2 in the sequence.  It is used when filling in reverse char part of value
		assert self.name == "wx2"
		if self.pvals["include_reverse_char"]:
			num_bins = len(self.pvals["bin_sizes"])
			if cindex <= num_bins:
				rev_part = self.pvals["empty_flag"]
			else:
				rev_bin_bits = address[self.pvals["last_char_position"]:self.pvals["wh_len"]]
				rev_char, hdist = self.env.cmap.bin2char(rev_bin_bits, nret = 1, match_bits=len(rev_bin_bits))[0]
				assert hdist == 0, "unable to lookup reverse character from last address bin, cindex=%s, hdist=%s" % (
					cindex, hdist)
				brev_char = self.env.cmap.char2bin(rev_char)
				rev_part = brev_char[0:self.pvals["len_reverse_part"]]
			fwd_part = bchar2[0:self.pvals["len_item_part"]]
			target_input = np.concatenate((fwd_part, rev_part))
		else:
			target_input = bchar2[0:self.pvals["len_item_part"]]
		new_value = np.concatenate((target_input, new_address[self.pvals["targets_len"]:]))
		assert len(new_value) == self.env.pvals["word_length"]
		return new_value


	def make_new_value_orig(self, address, new_address, bchar2, cindex):
		# bchar2 is item to store in value (next character in sequence)
		# cindex is 0-based index of bchar2 in the sequence.  It is used when filling in reverse char part of value
		assert self.name == "wx2"
		if self.pvals["include_reverse_char"]:
			num_bins = len(self.pvals["bin_sizes"])
			rev_part = (self.pvals["empty_flag"] if cindex <= num_bins else address[self.pvals["last_char_position"]:
					self.pvals["last_char_position"]+self.pvals["len_reverse_part"]])
			fwd_part = bchar2[0:self.pvals["len_item_part"]]
			item_input = np.concatenate((fwd_part, rev_part))
			assert len(item_input) == self.pvals["item_len"]
		else:
			item_input = bchar2[0:self.pvals["item_len"]]
		new_value = np.concatenate((item_input, new_address[self.pvals["item_len"]:]))
		assert len(new_value) == self.env.pvals["word_length"]
		return new_value

	def get_bchar0_part(self, address):
		# return bits of item_t0 (first bin of address in wx2 algorithm)
		return address[0:self.pvals["item_len"]]


	def make_reverse_address(self, address, value, char0, top_matches):
		# if cannot find character matching reverse char, include that in "top_matches"
		# char0 is char specified in first bin of address (most recent character in sequence)
		# shift bins left, insert bits from reverse char (in value), unless it matches empty_flag
		debug = self.env.pvals["debug"] == 1
		rev_part = value[self.pvals["len_item_part"]:self.pvals["targets_len"]]
		rev_top_matches = self.env.cmap.bin2char(rev_part, match_bits=len(rev_part))
		rev_char, hdist = rev_top_matches[0]
		if rev_char == "#":
			# stop going back, remaining characters of sequence are in value bins
			return None
		brev_char = self.env.cmap.char2bin(rev_char)
		item_len = self.pvals["item_len"]
		wh_len = self.pvals["wh_len"]
		hist_part = address[item_len:wh_len]
		bchar0 = self.env.cmap.char2bin(char0)
		xor_part = np.roll(np.bitwise_xor(address[wh_len:],bchar0[0:self.pvals["xr_len"]]),-1)
		reverse_address = np.concatenate((hist_part, brev_char[0:item_len], xor_part))
		return reverse_address

	def make_reverse_address_orig(self, address, value, char0, top_matches):
		# if cannot find character matching reverse char, include that in "top_matches"
		# char0 is char specified in first bin of address (most recent character in sequence)
		# shift bins left, insert bits from reverse char (in value), unless it matches empty_flag
		debug = self.env.pvals["debug"] == 1
		rev_part = value[self.pvals["len_item_part"]:self.pvals["item_len"]]
		rev_top_matches = self.env.cmap.bin2char(rev_part, match_bits=len(rev_part))
		rev_char, hdist = rev_top_matches[0]
		if rev_char == "#":
			# stop going back, remaining characters of sequence are in value bins
			return None
		brev_char = self.env.cmap.char2bin(rev_char)
		item_len = self.pvals["item_len"]
		wh_len = self.pvals["wh_len"]
		hist_part = address[item_len:wh_len]
		bchar0 = self.env.cmap.char2bin(char0)
		xor_part = np.roll(np.bitwise_xor(address[wh_len:],bchar0[0:self.pvals["xr_len"]]),-1)
		reverse_address = np.concatenate((hist_part, brev_char[0:item_len], xor_part))
		return reverse_address

	def extract_sequence_prefix(self, address):
		# extract prefix of sequence from bins in value
		debug = self.env.pvals["debug"] == 1
		num_bins = len(self.pvals["bin_sizes"])
		item_len = self.pvals["item_len"]
		found = ""
		threshold = int(item_len * self.env.pvals["char_match_fraction"])
		if debug:
			print("entered sequence_prefix, address=%s, threshold=%s" % (bina2str(address), threshold))
		for i in range(num_bins):
			top_matches = self.env.cmap.bin2char(address[i*item_len:(i+1)*item_len], match_bits=item_len)
			char, hdist = top_matches[0]
			if hdist > threshold:
				char = "?"
			if debug:
				print("bin %s, item=%s, top_matches=%s, char=%s" % (i, bina2str(address[i*item_len:(i+1)*item_len]),
					top_matches, char))
			found = char + found
		return found

	def make_cleaned_codes(self, chars):
		# return array containing codes that are in chars.  This is to update codes read from memory.
		assert "?" not in chars
		item_len = self.pvals["item_len"]
		cleaned_codes = np.full(item_len * len(chars), 0, dtype=np.int8)
		for i in range(len(chars)):
			char = chars[i]
			bchar = self.env.cmap.char2bin(char)
			cleaned_codes[i*item_len:(i+1)*item_len] = bchar[0:item_len]
		return cleaned_codes



	def make_shuffle_seed(self, address):
		# seed for shuffle made from first min_key_length bins
		mkl = self.env.pvals["min_key_length"]
		assert mkl > 1
		item_len = self.pvals["item_len"]
		pb_len = mkl * item_len
		pb = address[0:pb_len]
		# make seed by selecting first element of each row, then 2nd element of each row, ...
		# equivalent to converting from row major to column major order
		pb2d = np.reshape(pb, (mkl, item_len))
		seed = pb2d.flatten('f')
		return seed

	def make_shuffle_seed_orig(self, address):
		# seed for shuffle made from first two bins
		assert self.env.pvals["first_bin_fraction"] > 2, "seeded_shuffle option for wx2 requires at least 3 bins of equal size"
		item_len = self.pvals["item_len"]
		seed_part1 = address[0:item_len]
		seed_part2 = address[item_len:2*item_len]
		# following from https://stackoverflow.com/questions/17619415/how-do-i-combine-two-numpy-arrays-element-wise-in-python
		seed = np.insert(seed_part2, np.arange(item_len), seed_part1)
		return seed

	def wh_shuffle(self, address):
		if self.env.pvals["seeded_shuffle"] == 1:
			return self.wh_shuffle_v1(address)
		else:
			return self.wh_shuffle_v2(address)

	def wh_shuffle_v1(self, address):
		# scramble history bins (wh_part) of address using first two bins as seed
		seed = self.make_shuffle_seed(address)
		wh_len = self.pvals["wh_len"]
		wh_part = address[0:wh_len]
		xr_part = address[wh_len:]
		shuffled_wh = seeded_shuffle(wh_part, seed)
		shuffled_address = np.concatenate((shuffled_wh, xr_part))
		return shuffled_address

	def wh_shuffle_v2(self, address):
		# scramble history bins (wh_part) of address using first two bins as seed
		seed = self.make_shuffle_seed(address)
		return self.ku_shuffle(address, seed)


	def ku_shuffle(self, address, seed, inverse=False):
		# shuffle known part and unknown part with same seed
		wh_len = self.pvals["wh_len"]
		item_len = self.pvals["item_len"]
		mkl = self.env.pvals["min_key_length"]
		kk_len = mkl * item_len  # known key
		kk_part = address[0:kk_len]  # unknown key (char's might not be present)
		uk_part = address[kk_len:wh_len]
		shuffled_kk = seeded_shuffle(kk_part, seed, inverse)
		shuffled_uk = seeded_shuffle(uk_part, seed, inverse)
		xr_part = address[wh_len:]
		shuffled_address = np.concatenate((shuffled_kk, shuffled_uk, xr_part))
		return shuffled_address

class Ma_fl(Merge_algorithm):

	def initialize(self):
		# From Pentti: The first aN bits should be copied from the first aN bits of
		# the present vector, and the last bN bits should be copied
		# from the LAST bN bits of the permuted history vector.
		hf = self.env.pvals["history_fraction"] # Fraction of history to store when forming new address
		hist_len = int(hf * self.env.pvals["word_length"])
		assert hist_len > 0, "fl algorithm: must have hist_len(%s) > 0, hf (%s) is too small" % (hist_len, hf)
		item_len = self.env.pvals["word_length"] - hist_len
		assert item_len > 0, "fl algorithm: item_len must be > 0, is %s" % item_len
		self.pvals.update({"hist_len":hist_len, "item_len":item_len})
		self.pvals["shuffle_map"] = make_permutation_map(self.env.pvals["word_length"])

	def make_new_address(self, address, bchar1):
		part_1 = bchar1[0:self.pvals["item_len"]]
		part_2 = address[self.pvals["shuffle_map"]][-self.pvals["hist_len"]:]
		new_address = np.concatenate((part_1, part_2))
		assert len(new_address) == self.env.pvals["word_length"], "fl algorithm, len(address) (%s) != word_length (%s)" % (
			len(new_address), self.env.pvals["word_length"])
		return new_address

class Ma_hh(Merge_algorithm):

	def make_new_address(self, address, bchar1):
		# original merge, select every other bit from each and concatinate
		# should be same as wx with xor_fraction == 0 and history_fraction = 0.5
		new_address = np.concatenate((bchar1[0::2],address[0::2]))
		assert len(new_address) == self.env.pvals["word_length"]
		return new_address

class Ma_hh2(Merge_algorithm):

	def make_new_address(self, address, bchar1):
		# should be more exact match to wx with xor_fraction == 0 and history_fraction = 0.5
		start_bit = self.env.pvals["start_bit"]
		word_length = self.env.pvals["word_length"]
		new_address = np.concatenate((bchar1[0:int(word_length/2)], address[start_bit::2]))
		assert len(new_address) == word_length
		return new_address


class Char_map:
	# maps each character in a sequence to a random binary word
	# below specifies maximum number of unique characters
	max_unique_chars = 50

	def __init__(self, seq, word_length = 128, debug=False):
		# seq - a Sequence object
		# word_length - number of bits in random binary word
		# repeat - make second half of each word same as first half (used for wx2)
		assert word_length % 2 == 0, "word_length must be even"
		self.word_length = word_length
		self.chars = ''.join(sorted(list(set(list(''.join(seq))))))  # sorted string of characters appearing in seq
		self.chars += "#"  # stop char - used to indicate end of sequence
		self.check_length()
		self.debug = debug
		self.binary_vals = initialize_binary_matrix(self.max_unique_chars, word_length, force_unique=debug)
		if debug:
			self.show_codes()

	def show_codes(self):
		for i in range(len(self.chars)):
			print("char '%s' is %s " % (self.chars[i], bina2str(self.binary_vals[i])))
	
	def check_length(self):
		# make sure not too many unique characters 
		if len(self.chars) > self.max_unique_chars:
			sys.exit("Too many unique characters (%s) max is %s.  Increase constant: max_unique_chars" % (
				len(self.chars), self.max_unique_chars))

	def char2bin(self, char):
		# return binary word associated with char
		index = self.chars.find(char)
		if index == -1:
			# this is a new character, add it to chars string
			index = len(self.chars)
			self.chars += char
			self.check_length()
			if self.debug:
				print("char '%s' is %s " % (char, bina2str(self.binary_vals[index])))
		return self.binary_vals[index]

	def bin2char(self, b, nret = 3, match_bits=None):
		# returns array of top matching characters and the hamming distance (# of bits not matching)
		# b is a binary array, same length as word_length
		# nret is the number of matches to return
		# if not None, match_bits is number of bits to match
		top_matches = find_matches(self.binary_vals[0:len(self.chars),:], b, nret, match_bits=match_bits)
		top_matches = [ (self.chars[x[0]], x[1]) for x in top_matches ]
		return top_matches

	def part2full(self, b, match_bits=None):
		# return full length binary array corresponding to first b bits of b.  match_bits is number
		# of bits in b to use for matching.
		char = self.bin2char(b, nret = 1, match_bits=match_bits)[0][0]
		full = self.char2bin(char)
		return full


class Sdm:
	# implements a sparse distributed memory
	def __init__(self, address_length=128, word_length=128, num_rows=512, nact=5, debug=False):
		# nact - number of active addresses used (top matches) for reading or writing
		self.address_length = address_length
		self.word_length = word_length
		self.num_rows = num_rows
		self.nact = nact
		self.data_array = np.zeros((num_rows, word_length), dtype=np.int8)
		self.addresses = initialize_binary_matrix(num_rows, word_length)
		self.debug = debug
		self.hits = np.zeros((num_rows,), dtype=np.int32)

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		d = data.copy()
		d[d==0] = -1  # replace zeros in data with -1
		for i in top_matches:
			self.data_array[i] += d
			self.hits[i] += 1
		if self.debug:
			print("store\n addr=%s\n data=%s" % (bina2str(address), bina2str(data)))

	def show_hits(self):
		# display histogram of overlapping hits
		values, counts = np.unique(self.hits, return_counts=True)
		vc = [(values[i], counts[i]) for i in range(len(values))]
		vc.sort(key = lambda y: (y[0], y[1]))
		print("hits - counts:")
		pp.pprint(vc)

	def read(self, address, match_bits=None):
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True, match_bits=match_bits)
		i = top_matches[0]
		sum = np.int32(self.data_array[i].copy())  # np.int32 is to convert to int32 to have range for sum
		for i in top_matches[1:]:
			sum += self.data_array[i]
		sum[sum<1] = 0   # replace values less than 1 with zero
		sum[sum>0] = 1   # replace values greater than 0 with 1
		if self.debug:
			print("read\n addr=%s\n top_matches=%s\n data=%s" % (bina2str(address), top_matches, bina2str(sum)))
		return sum

	def clear(self):
		# set data_array contents to zero
		self.data_array.fill(0)

def bina2str(address):
	# convert binary array address to a string
	if address is None:
		return None
	binary_string = "".join(["%s" % i for i in address])
	# following from: https://stackoverflow.com/questions/2072351/python-conversion-from-binary-string-to-hexadecimal
	hex_string = '%0*x' % ((len(binary_string) + 3) // 4, int(binary_string, 2))
	wlength = 8  # insert space between each 8 characters.  From: 
	# https://stackoverflow.com/questions/10070434/how-do-i-insert-a-space-after-a-certain-amount-of-characters-in-a-string-using-p
	# binary_string_with_spaces = ' '.join(binary_string[i:i+wlength] for i in range(0,len(binary_string),wlength))
	hex_string_with_spaces = ' '.join(hex_string[i:i+wlength] for i in range(0,len(hex_string),wlength))
	return hex_string_with_spaces


def hamming(b1, b2):
	# compute hamming distance
	assert len(b1) == len(b2)
	ndiff = np.count_nonzero(b1!=b2)
	return ndiff

# def converge_from_fixed_seed(env, address, fixed_address, change_mask, max_steps):
# 	hconverge=[]
# 	prev_address = address
# 	best_hdiff = word_length
# 	while True:
# 		found_value = env.sdm.read(address)
# 		address = np.bitwise_or(fixed_address, np.bitwise_and(found_value, chng_mask))
# 		# address = np.concatenate((address[0:pb_len], found_value[pb_len:]))
# 		hdiff = hamming(prev_address, address)
# 		if hdiff < best_hdiff:
# 			best_address = address
# 			best_found_value = found_value
# 		hconverge.append(hdiff)
# 		f_non_zero = np.count_nonzero(found_value) / len(found_value)
# 		if hdiff == 0:
# 			print("Seed %s,iterative converged in %s steps, f_non_zero=%s, hdiff=%s" % (seed_count,
# 				len(hconverge), f_non_zero, hconverge))
# 			break
# 		if len(hconverge) > max_steps:
# 			print("Seed %s, did not converge in %s steps, f_non_zero=%s, hdiff=%s" % (seed_count, len(hconverge),
# 				f_non_zero, hconverge))
# 			if seed_count >= max_seeds:
# 				print("Converge failed after %s seeds" % seed_count)
# 				break
# 			seed_count += 1
# 			hconverge=[]
# 			# make a new seed for next convergence attempt
# 			seed = np.bitwise_xor(np.roll(seed, 1), seed)
# 			# address =  np.concatenate((address[0:pb_len], seed[pb_len:]))
# 			address = np.bitwise_or(fixed_address, np.bitwise_and(seed, chng_mask))
# 		prev_address = address


def converge(env, address, pb_len, shuf):
	# converge address by reapeat reading bits after pb_len match bits
	# if shuf is True, shuffle wh_part of address for convergence, then unshuffle
	word_length = env.pvals["word_length"]
	assert len(address) == word_length
	wh_len = env.ma.pvals["wh_len"]
	item_len = env.ma.pvals["item_len"]
	hconverge=[]
	seed_count = 1
	max_steps, max_seeds = map( int, env.pvals["converge_count"].split(',') )
	seed = address
	best_hdiff = word_length
	if shuf:
		# make mask to indicate which bits are known, which are not
		# also shuffle wh part of address
		if env.pvals["seeded_shuffle"] == 1:
			# version 1 of seeded shuffle (I think does not work correctly)
			shuf_seed = env.ma.make_shuffle_seed(address)
			wh_len = env.ma.pvals["wh_len"]
			xr_len = env.ma.pvals["xr_len"]
			wh_mask = np.full(wh_len, 0, dtype=np.int8)
			wh_mask[0:pb_len] = 1
			wh_mask = seeded_shuffle(wh_mask, shuf_seed)
			xr_mask = np.full(xr_len, 0, dtype=np.int8)
			keep_mask = np.concatenate((wh_mask, xr_mask))
			wh_part = seeded_shuffle(address[0:wh_len], shuf_seed)
			xr_part = address[wh_len:]
			address = np.concatenate((wh_part, xr_part))
		else:
			# version 2 - new and improved
			shuf_seed = env.ma.make_shuffle_seed(address)
			wh_len = env.ma.pvals["wh_len"]
			xr_len = env.ma.pvals["xr_len"]
			wh_mask = np.full(wh_len, 0, dtype=np.int8)
			wh_mask[0:pb_len] = 1
			xr_mask = np.full(xr_len, 0, dtype=np.int8)
			keep_mask = env.ma.ku_shuffle(np.concatenate((wh_mask, xr_mask)), shuf_seed)
			address = env.ma.ku_shuffle(address, shuf_seed)
		assert len(keep_mask) == word_length
		assert len(address) == word_length
	else:
		# no shuffle, known_mask is just covering first pb_len bits
		keep_mask = np.full(word_length, 0, dtype=np.int8)
		keep_mask[0:pb_len] = 1
	chng_mask = 1 - keep_mask
	fixed_address = np.bitwise_and(address, keep_mask)
	prev_address = address
	froze_wh = True if pb_len == wh_len else False
	pre_clear_xor = None
	while True:
		found_value = env.sdm.read(address)
		address = np.bitwise_or(fixed_address, np.bitwise_and(found_value, chng_mask))
		# address = np.concatenate((address[0:pb_len], found_value[pb_len:]))
		hdiff = hamming(prev_address, address)
		if hdiff < best_hdiff:
			if pre_clear_xor is not None:
				print("updated best_address after froze_wh")
			best_address = address
			best_found_value = found_value
			best_hdiff = hdiff
		hconverge.append(hdiff)
		f_non_zero = np.count_nonzero(found_value) / len(found_value)
		if hdiff == 0:
			print("Seed %s,iterative converged in %s steps, f_non_zero=%s, hdiff=%s" % (seed_count,
				len(hconverge), f_non_zero, hconverge))
			if not froze_wh:
				# now replace items beyond pb_len (read from value) with their cleaned up value (from item memory)
				address = env.ma.ku_shuffle(best_address, shuf_seed, inverse=True)
				found_sequence = env.ma.extract_sequence_prefix(address)
				found_sequence = found_sequence[::-1]  # reverse order
				if "?" in found_sequence:
					print("Unable to find all chars in sequence, aborting cleaning codes: %s" % found_sequence)
					break
				prefix_len = int(pb_len / item_len)
				cleaned_codes = env.ma.make_cleaned_codes(found_sequence[prefix_len:])
				assert len(cleaned_codes) == wh_len - pb_len, ("len(cleaned_codes):%s != wh_len (%s) - pb_len(%s)"
					" prefix_len=%s, item_len=%s, found_sequence='%s'" % (
					len(cleaned_codes), wh_len, pb_len, prefix_len, item_len, found_sequence))
				hcc = hamming(address[pb_len:wh_len], cleaned_codes)
				if hcc == 0:
					print("No change needed after finding cleaned_codes")
					break
				else:
					print("cleaned codes changes found.  Sequence='%s' hamming distance=%s, continuing convege" % (
						found_sequence, hcc))
					hconverge = [ hcc ]
					assert len(cleaned_codes) == wh_len - pb_len
					address[pb_len:wh_len] = cleaned_codes
					address = env.ma.ku_shuffle(address, shuf_seed)
					hcheck = hamming(address, best_address)
					print("updating address, hcheck=%s" % hcheck)
					keep_mask = np.full(word_length, 0, dtype=np.int8)
					keep_mask[0:wh_len] = 1
					chng_mask = 1 - keep_mask
					fixed_address = np.bitwise_and(address, keep_mask)
					froze_wh = True
					best_hdiff = hcc
					pre_clear_xor = address[wh_len:]
			else:
				if pre_clear_xor is not None:
					adr_chng = hamming(pre_clear_xor, best_address[wh_len:])
					print("Done converge after fixing wh.  Change in address is: %s" % adr_chng)
				break
		if len(hconverge) > max_steps:
			print("Seed %s, did not converge in %s steps, f_non_zero=%s, hdiff=%s" % (seed_count, len(hconverge),
				f_non_zero, hconverge))
			if pre_clear_xor is not None:
				adr_chng = hamming(pre_clear_xor, best_address[wh_len:])
				print("Stoping converge after fixing wh.  Change in address is: %s" % adr_chng)
				break
			if seed_count >= max_seeds:
				print("Converge failed after %s seeds" % seed_count)
				break
			seed_count += 1
			hconverge=[]
			# make a new seed for next convergence attempt
			seed = np.bitwise_xor(np.roll(seed, 1), seed)
			# address =  np.concatenate((address[0:pb_len], seed[pb_len:]))
			address = np.bitwise_or(fixed_address, np.bitwise_and(seed, chng_mask))
		prev_address = address
	if shuf:
		# need to unshuffle address before returning
		if env.pvals["seeded_shuffle"] == 1:
			wh_part = seeded_shuffle(best_address[0:wh_len], shuf_seed, inverse=True)
			xr_part = best_address[wh_len:]
			best_address = np.concatenate((wh_part, xr_part))
		else:
			# version 2
			best_address = env.ma.ku_shuffle(best_address, shuf_seed, inverse=True)
	found_sequence = env.ma.extract_sequence_prefix(best_address)
	print("Found chars in initial address = %s" % found_sequence)
	return [best_address, f_non_zero, best_found_value]

def converge_orig(env, address, pb_len, shuf):
	# converge address by reapeat reading bits after pb_len match bits
	# if shuf is True, shuffle wh_part of address for convergence, then unshuffle
	word_length = env.pvals["word_length"]
	assert len(address) == word_length
	hconverge=[]
	seed_count = 1
	max_steps, max_seeds = map( int, env.pvals["converge_count"].split(',') )
	seed = address
	best_hdiff = word_length
	if shuf:
		# make mask to indicate which bits are known, which are not
		# also shuffle wh part of address
		if env.pvals["seeded_shuffle"] == 1:
			# version 1 of seeded shuffle (I think does not work correctly)
			shuf_seed = env.ma.make_shuffle_seed(address)
			wh_len = env.ma.pvals["wh_len"]
			xr_len = env.ma.pvals["xr_len"]
			wh_mask = np.full(wh_len, 0, dtype=np.int8)
			wh_mask[0:pb_len] = 1
			wh_mask = seeded_shuffle(wh_mask, shuf_seed)
			xr_mask = np.full(xr_len, 0, dtype=np.int8)
			keep_mask = np.concatenate((wh_mask, xr_mask))
			wh_part = seeded_shuffle(address[0:wh_len], shuf_seed)
			xr_part = address[wh_len:]
			address = np.concatenate((wh_part, xr_part))
		else:
			# version 2 - new and improved
			shuf_seed = env.ma.make_shuffle_seed(address)
			wh_len = env.ma.pvals["wh_len"]
			xr_len = env.ma.pvals["xr_len"]
			wh_mask = np.full(wh_len, 0, dtype=np.int8)
			wh_mask[0:pb_len] = 1
			xr_mask = np.full(xr_len, 0, dtype=np.int8)
			keep_mask = env.ma.ku_shuffle(np.concatenate((wh_mask, xr_mask)), shuf_seed)
			address = env.ma.ku_shuffle(address, shuf_seed)
		assert len(keep_mask) == word_length
		assert len(address) == word_length
	else:
		# no shuffle, known_mask is just covering first pb_len bits
		keep_mask = np.full(word_length, 0, dtype=np.int8)
		keep_mask[0:pb_len] = 1
	chng_mask = 1 - keep_mask
	fixed_address = np.bitwise_and(address, keep_mask)
	prev_address = address
	while True:
		found_value = env.sdm.read(address)
		address = np.bitwise_or(fixed_address, np.bitwise_and(found_value, chng_mask))
		# address = np.concatenate((address[0:pb_len], found_value[pb_len:]))
		hdiff = hamming(prev_address, address)
		if hdiff < best_hdiff:
			best_address = address
			best_found_value = found_value
		hconverge.append(hdiff)
		f_non_zero = np.count_nonzero(found_value) / len(found_value)
		if hdiff == 0:
			print("Seed %s,iterative converged in %s steps, f_non_zero=%s, hdiff=%s" % (seed_count,
				len(hconverge), f_non_zero, hconverge))
			break
		if len(hconverge) > max_steps:
			print("Seed %s, did not converge in %s steps, f_non_zero=%s, hdiff=%s" % (seed_count, len(hconverge),
				f_non_zero, hconverge))
			if seed_count >= max_seeds:
				print("Converge failed after %s seeds" % seed_count)
				break
			seed_count += 1
			hconverge=[]
			# make a new seed for next convergence attempt
			seed = np.bitwise_xor(np.roll(seed, 1), seed)
			# address =  np.concatenate((address[0:pb_len], seed[pb_len:]))
			address = np.bitwise_or(fixed_address, np.bitwise_and(seed, chng_mask))
		prev_address = address
	if shuf:
		# need to unshuffle address before returning
		if env.pvals["seeded_shuffle"] == 1:
			wh_part = seeded_shuffle(best_address[0:wh_len], shuf_seed, inverse=True)
			xr_part = best_address[wh_len:]
			best_address = np.concatenate((wh_part, xr_part))
		else:
			# version 2
			best_address = env.ma.ku_shuffle(best_address, shuf_seed, inverse=True)
	found_sequence = env.ma.extract_sequence_prefix(best_address)
	print("Found chars in initial address = %s" % found_sequence)
	return [best_address, f_non_zero, best_found_value]

def try_alternate(env, prev_address, alt_chars, shuf, reverse, xr_thresh):
	# try alternate characters for continuing sequence
	assert reverse is False, "try_alternate not implemented for reverse recall"
	wh_len = env.ma.pvals["wh_len"]
	for ch in alt_chars:
		assert not isinstance(ch, str), "expecting tuple, found str: %s" % ch
		char, hdist = ch
		bchar = env.cmap.char2bin(char)
		address = env.ma.make_new_address(prev_address, bchar)
		shuffle_address = env.ma.wh_shuffle(address) if shuf else address
		value = env.sdm.read(shuffle_address)
		recalled_xor_part = value[wh_len:]
		address_xor_part = address[wh_len:]
		xor_hamming = hamming(recalled_xor_part,address_xor_part)
		if xor_hamming < xr_thresh:
			msg = "fixed to '%s'" % char
			return (address, value, char, msg)
	# didn't find alternate
	msg = "alt search failed."
	return (None, None, None, msg)


def recall(prefix, env, reverse=False):
	# recall sequence starting with prefix
	# build address to start searching
	# reverse is True to search in reverse
	if reverse and len(prefix) == 1:
		return [ ["Error: cannot reverse recall with one char",], prefix]
	word_length = env.pvals["word_length"]
	max_num_recalled_chars = len(max(env.saved_strings, key=len)) + 5
	ma = env.pvals["merge_algorithm"]
	if ma == "wx2":
		wh_len = env.ma.pvals["wh_len"]
		item_len = env.ma.pvals["item_len"]
		xr_len = env.ma.pvals["xr_len"]
		prev_address = None
		xr_thresh = int(env.pvals["char_match_fraction"] * xr_len)
		rev_thresh = int(env.pvals["char_match_fraction"] * item_len)
	shuf = ma == "wx2" and env.pvals["seeded_shuffle"] > 0
	rfix = ma == "wx2" and env.pvals["recall_fix"] == 1
	debug = env.pvals["debug"] == 1
	# create initial address from prefix
	bchar = env.cmap.char2bin(prefix[0])
	address = env.ma.make_initial_address(bchar)
	for char in prefix[1:]:
		bchar = env.cmap.char2bin(char)
		address = env.ma.make_new_address(address, bchar)
	if debug:
		print("prefix[0]='%s'\nbchar='%s'\naddress='%s'" % (prefix[0], bina2str(bchar), bina2str(address)))

	# address = env.cmap.char2bin(prefix[0])
	# for char in prefix:
	# 	address = env.ma.make_new_address(address, env.cmap.char2bin(char))
	if not reverse:
		found = [ prefix, ]
		word2 = prefix
	else:
		# for reverse, don't include prefix since it will be recalled in reverse
		found = []
		word2 = ""
	if ma  == "wx2" and len(prefix) > 1:
		# iteratively read value at address until converges (no decrease in hamming distance between xor part in
		# address and read value)
		pb_len = sum(env.ma.pvals["bin_sizes"][0:len(prefix)]) # number prefix bits
		# max_converge_trys = 10
		address, f_non_zero, found_value = converge(env, address, pb_len, shuf)
		if f_non_zero < 0.1:
			print("Failed to converge to valid data (found_value near zero)")
			return [found, word2]
		if debug:
			print(" found_value = %s" % bina2str(found_value))
			print(" created address used to start reading sequence:\n %s" % bina2str(address))
	# now read sequence using address derived from prefix
	while True:
		msg = []
		shuffle_address = env.ma.wh_shuffle(address) if shuf else address
		value = env.sdm.read(shuffle_address)
		if ma == "wx2":
			recalled_xor_part = value[wh_len:]
			address_xor_part = address[wh_len:]
			xor_hamming = hamming(recalled_xor_part,address_xor_part)
			msg.append("xorh=%s" % xor_hamming)
			if rfix:
				if not reverse and prev_address is not None and xor_hamming > xr_thresh:
					# hamming distance is greater than threshold, backtrack and try alternate characters
					alt_address, alt_value, alt_char, alt_msg = try_alternate(env, prev_address, found[-1][1:-1], shuf, reverse, xr_thresh)
					if alt_address is not None:
						address = alt_address
						value = alt_value
						word2 = word2[0:-1] + alt_char
					msg.append(">xr_thresh(%s): %s" % (xr_thresh, alt_msg))
				# elif reverse and hamming(address[env.ma.pvals["last_char_position"]:wh_len],
				# 	value[env.ma.pvals["last_char_position"]:wh_len]) > rev_thresh:
				# 	msg.append(">rev_thres(%s) - fix not yet implemented" % rev_thresh)
		new_address, found_char, top_matches = env.ma.prev(address, value) if reverse else env.ma.next(address, value)
		found.append(top_matches)
		if reverse:
			word2 = found_char + word2
		if new_address is None or len(word2) > max_num_recalled_chars:
			break
		if not reverse:
			word2 += found_char
		diff = np.count_nonzero(address!=new_address)
		msg.insert(0,"addr_diff=%s" % diff)  # insert at beginning of messages
		if diff == 0:
			msg.append("found fixed point in address")
			break
		if ma == "wx2":
			# recalled_xor_part = value[wh_len:]
			# address_xor_part = address[wh_len:]
			# xor_hamming = "xorh=%s" % hamming(recalled_xor_part,address_xor_part)
			prev_address = address
			prev_value = value
		address = new_address
		if msg:
			found[-1].append(", ".join(msg))
			msg = []
	if msg:
		found[-1].append(", ".join(msg))
	found2 = []
	for item in found:
		if isinstance(item, str):
			found2.append(item)
		else:
			chars = " ".join(["'%s'%s" % (p[0], p[1]) for p in item[0:-1]])
			msgs = " %s" % item[-1] if len(item) > 1 and isinstance(item[-1], str) else ""
			found2.append("%s%s" % (chars, msgs))
	return [found2, word2]


def recall_orig(prefix, env, reverse=False):
	# recall sequence starting with prefix
	# build address to start searching
	# reverse is True to search in reverse
	if reverse and len(prefix) == 1:
		return [ ["Error: cannot reverse recall with one char",], prefix]
	word_length = env.pvals["word_length"]
	max_num_recalled_chars = len(max(env.saved_strings, key=len)) + 5
	ma = env.pvals["merge_algorithm"]
	if ma == "wx2":
		wh_len = env.ma.pvals["wh_len"]
		item_len = env.ma.pvals["item_len"]
	shuf = ma == "wx2" and env.pvals["seeded_shuffle"] == 1
	debug = env.pvals["debug"] == 1
	# create initial address from prefix
	bchar = env.cmap.char2bin(prefix[0])
	address = env.ma.make_initial_address(bchar)
	for char in prefix[1:]:
		bchar = env.cmap.char2bin(char)
		address = env.ma.make_new_address(address, bchar)
	if debug:
		print("prefix[0]='%s'\nbchar='%s'\naddress='%s'" % (prefix[0], bina2str(bchar), bina2str(address)))

	# address = env.cmap.char2bin(prefix[0])
	# for char in prefix:
	# 	address = env.ma.make_new_address(address, env.cmap.char2bin(char))
	if not reverse:
		found = [ prefix, ]
		word2 = prefix
	else:
		# for reverse, don't include prefix since it will be recalled in reverse
		found = []
		word2 = ""
	if ma  == "wx2" and len(prefix) > 1:
		# iteratively read value at address until converges (no decrease in hamming distance between xor part in
		# address and prev_address)
		pb_len = sum(env.ma.pvals["bin_sizes"][0:len(prefix)]) # number prefix bits
		# max_converge_trys = 10
		address, f_non_zero, found_value = converge(env, address, pb_len, shuf)
		if f_non_zero < 0.1:
			print("Failed to converge to valid data (found_value near zero)")
			return [found, word2]
		if debug:
			print(" found_value = %s" % bina2str(found_value))
			print(" created address used to start reading sequence:\n %s" % bina2str(address))
	computed_address = address
	# now read sequence using address derived from prefix
	while True:
		shuffle_address = env.ma.wh_shuffle(address) if shuf else address
		value = env.sdm.read(shuffle_address)
		new_address, found_char, top_matches = env.ma.prev(address, value) if reverse else env.ma.next(address, value)
		found.append(top_matches)
		if reverse:
			word2 = found_char + word2
		if new_address is None or len(word2) > max_num_recalled_chars:
			break
		if not reverse:
			word2 += found_char
		diff = np.count_nonzero(address!=new_address)
		found[-1].append("addr_diff=%s" % diff)
		if diff == 0:
			found[-1].append("found fixed point in address")
			break
		if ma == "wx2":
			assert hamming(address, computed_address) == 0
			recalled_xor_part = value[wh_len:]
			address_xor_part = address[wh_len:]
			computed_xor_part = computed_address[wh_len:]
			xor_hamming = " xorh cv=%s, av=%s" % (
				hamming(computed_xor_part, recalled_xor_part), hamming(recalled_xor_part,address_xor_part))
			found[-1].append(xor_hamming)
			bchar = env.cmap.char2bin(found_char)
			if not reverse:
				computed_address = env.ma.make_new_address(computed_address, bchar)
			else:
				computed_address = env.ma.make_reverse_address(computed_address, value, found_char, top_matches)
		address = new_address
	return [found, word2]

def store_strings(env, strings):
	# debug = env.pvals["debug"] == 1
	ma = env.pvals["merge_algorithm"]
	shuf = ma == "wx2" and env.pvals["seeded_shuffle"] > 0
	debug = env.pvals["debug"] == 1
	if shuf:
		wh_len = env.ma.pvals["wh_len"]
	for string in strings:
		print("storing '%s'" % string)
		address = env.ma.make_initial_address(env.cmap.char2bin(string[0]))
		bchar1 = env.cmap.char2bin(string[1])
		shuffled_address = env.ma.wh_shuffle(address) if shuf else address
		value = env.ma.make_initial_value(address, bchar1)
		if debug:
			print("string[0]='%s', string[1]='%s'\naddress='%s', bchar1='%s'\nshufadd='%s'\n  value='%s'" %
				(string[0], string[1], bina2str(address), bina2str(bchar1), bina2str(shuffled_address), bina2str(value)))
		env.sdm.store(shuffled_address, value)
		strpstop = string+"#"
		for cindex in range(2, len(strpstop)):
			bchar2 = env.cmap.char2bin(strpstop[cindex])
			new_address = env.ma.make_new_address(address, bchar1)
			shuffled_address = env.ma.wh_shuffle(new_address) if shuf else new_address
			new_value = env.ma.make_new_value(address, shuffled_address, bchar2, cindex)
			env.sdm.store(shuffled_address, new_value)
			bchar1 = bchar2
			address = new_address
		env.record_saved_string(string)


def test_merge_convergence(env):
	# test how fast address for substring converges to address for full string
	string = "abcdefghijklmnopqrstuvwxyz0123456789"
	address = env.ma.make_initial_address(env.cmap.char2bin(string[-2]))
	subaddr = env.ma.make_initial_address(env.cmap.char2bin(string[-1]))
	result = ""
	ccount = 0
	while ccount <= (env.pvals["word_length"]+2):
		char = string[ccount % len(string)]
		bchar = env.cmap.char2bin(char)
		address = env.ma.make_new_address(address, bchar)
		subaddr = env.ma.make_new_address(subaddr, bchar)
		distance = hamming(address, subaddr)
		result+="%s-%s " %(ccount+1, distance)
		if distance == 0:
			break
		ccount += 1
		if ccount % 25 == 0:
			result+="\n"
	print("Convergence for merge %s, hf=%s, xf=%s, fbf=%s, start_bit=%s:\n%s" % (env.pvals["merge_algorithm"],
		env.pvals["history_fraction"], env.pvals["xor_fraction"], env.pvals["first_bin_fraction"],
		env.pvals["start_bit"], result))


def store_param_strings(env):
	# store strings provided as parameters in command line (or updated)
	store_strings(env, env.pvals["string_to_store"])


def recall_strings(env, strings, prefix_mode = False, reverse = False):
	# recall list of strings starting with first character of each
	# if prefix_mode is True, use entire string as prefix for recall
	if len(env.saved_strings) < len(strings):
		print("Cannot recall %s strings because %s strings are stored" % (len(strings), len(env.saved_strings)))
		return
	if reverse:
		if not prefix_mode:
			print("reverse recall not allowed with just first char of string")
			return
		if env.pvals["merge_algorithm"] != "wx2" or not env.ma.pvals["include_reverse_char"]:
			print("Reverse recall is not enabled.  Requries: wx2 merge_algorithm, "
				"first_bin_fraction > 1 and allow_reverse_recall == 1")
			return
	error_count = 0
	for word in strings:
		print ("\nRecall '%s'" % word)
		prefix = word if prefix_mode else word[0]
		found, word2 = recall(prefix, env, reverse)
		if word != word2:
			error_count += 1
			msg = "ERROR"
		else:
		    msg = "Match"
		pp.pprint(found)
		print ("Recall '%s'" % word)
		if not prefix_mode:
			print("found: '%s' - %s" % (word2, msg))
		else:
			print("found: '%s'" % word2)
	if not prefix_mode:
		print("%s strings, %s errors" % (len(strings), error_count))

def recall_param_strings(env):
	# recall stored sequences starting with first character
	recall_strings(env, env.pvals["string_to_store"])

def main():
	env = Env()
	store_param_strings(env)
	# retrieve sequences, starting with first character
	recall_param_strings(env)
	do_interactive_commands(env)

main()
