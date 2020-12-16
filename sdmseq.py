# SDM model test simulation
import numpy as np
from operator import itemgetter
import pprint
pp = pprint.PrettyPrinter(indent=4)
import argparse
import shlex
import readline
readline.parse_and_bind('tab: complete')
# import sys


class Env:
	# stores environment settings and data arrays

	# command line arguments, also used for parsing interactive update of parameters
	parms = [
		{ "name":"word_length", "kw":{"help":"Word length for address and memory", "type":int},
	 	  "flag":"w", "required_init":"i", "default":256 },
	 	{ "name":"num_rows", "kw":{"help":"Number rows in memory","type":int},
	 	  "flag":"r", "required_init":"i", "default":512 },
	 	{ "name":"activation_count", "kw":{"help":"Number memory rows to activate for each address","type":int},
	 	  "flag":"a", "required_init":"m", "default":5},
	 	{ "name":"char_match_fraction", "kw": {"help":"Fraction of word_length to form hamming distance threshold for"
			" matching character to item memory","type":float},"flag":"cmf", "required_init":"", "default":0.25},
		{ "name":"merge_algorithm", "kw":{"help":"Algorithm used combine item and history when forming new address. "
		    "wx - Weighted and/or XOR, wx2 - save xor with data, fl - First/last bits, hh - concate every other bit",
		    "choices":["wx", "wx2", "fl", "hh", "hh2"]},
	 	  "flag":"ma", "required_init":"m", "default":"hh2"},
		# { "name":"permute", "kw":{"help":"Permute values when storing","type":int, "choices":[0, 1]},
		# 	  "flag":"p", "required_init":True, "default":0},
		{ "name":"start_bit", "kw":{"help":"Starting bit for debugging","type":int, "choices":[0, 1]},
		  "flag":"b", "required_init":"m", "default":0},
	 	{ "name":"history_fraction", "kw":{"help":"Fraction of history to store when forming new address."
	 	    "(Both wx and fl algorighms)", "type":float}, "flag":"hf", "required_init":"m", "default":0.5},
	 	{ "name":"xor_fraction", "kw":{"help":"Fraction of bits used for xor component in wx algorithm","type":float},
	 	  "flag":"xf", "required_init":"m", "default":0.5},
	 	{ "name":"debug", "kw":{"help":"Debug mode","type":int, "choices":[0, 1]},
		   "flag":"d", "required_init":"", "default":0},
		{ "name":"string_to_store", "kw":{"help":"String to store","type":str,"nargs":'*'}, "required_init":"",
		  "flag":"s", "default":
		  '"happy day" "evans hall" "campanile" "sutardja dai hall" "oppenheimer"'
		  ' "distributed memory" "abcdefghijklmnopqrstuvwxyz"'
		  # ' "Evans rrrrr math" "Corey rrrrr eecs"'
		  }]

	def __init__(self):
		# self.state = {"initialized": False, "initialized_required": False, "cleared": False, "cleared_required": False}
		# self.state = {"initialize_needed": False, "clear_": False, "cleared": False, "cleared_required": False}
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
		# initialize sdm and char_map
		word_length = self.pvals["word_length"]
		# if not required_init:
		# 	self.cmap.re
		print("Initializing.")
		self.cmap = Char_map(self.pvals["string_to_store"], word_length=word_length)
		self.sdm = Sdm(address_length=word_length, word_length=word_length, num_rows=self.pvals["num_rows"],
			debug=self.pvals["debug"])
		self.merge = Merge(self)
		self.saved_strings = []
		# self.state["initialized"] = True
		# self.state["initialized_required"] = False
		self.required_init = ""  # contains chars: 'i' - initialize structures, 'm' - clear memory, '' - none-needed

	def record_saved_string(self, string):
		self.saved_strings.append(string)

	def ensure_initialized(self):
		# make sure initialized or memory cleared after parameter changes
		if "i" in self.required_init:
			self.initialize()
		elif "m" in self.required_init:
			self.clear()
			self.merge.initialize()
		else:
			assert self.required_init == "", "invalid char in required_init: %s" % self.required_init

	# def ensure_cleared(self):
	# 	elif 'c' in self.required_init and len(self.saved_strings) > 0:
	# 		self.clear()

	def clear(self):
		if 'i' in self.required_init:
			# must initialize rather than clear
			print("Initialization required before clearing.")
			self.initialize()
		else:
			print("Clearing memory.")
			self.sdm.clear()
			self.merge
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
		except argparse.ArgumentError:
			print('Invalid entry, try again.')
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
		" u - update parameters\n"
		" c - clear memory\n"
		" t - test merge convergence\n"
		" i - initalize everything")
	print(instructions)
	while True:
		try:
			line=input("> ")
		except EOFError:
			break;
		if len(line) == 0 or line[0] not in "sruict":
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
		# remaining commands may requre reinitializing or memory clearing if parameter settings have changed 
		env.ensure_initialized()
		if cmd == "r" and len(arg) > 0:
			print("recall prefix %s" % arg)
			recall_strings(env, shlex.split(arg), prefix_mode=True)
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


# class Sequences:
# 	# sequences learned
# 	def __init__(self, seq = ["happy_day", "evans_hall", "campanile", "sutardja_dai_hall", "oppenheimer"]):
# 		self.seq = seq


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
	hamming = matches[nret-1][1]
	nh = 1
	while matches[nret-1+nh][1] == hamming:
		if matches[nret-2+nh][0] > matches[nret-1+nh][0]:
			sys.exit("found matches with index out of order")
		nh += 1
	# if nh > 1:
	# 	print("Found %s  match last hamming: %s" % (nh, matches[nret-1:nret-1+nh]))
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

class Merge():
	# functions for different types of merges (combining history and new vector)

	# { "name":"merge_algorithm", "kw":{"help":"Algorithm used combine item and history when forming new address. "
	#     "wx - Weighted and/or XOR, fl - First/last bits", "choices":["wx","fl"]},
	# 	  "flag":"ma", "required_init":True, "default":"wx"},
	# 	{ "name":"history_fraction", "kw":{"help":"Fraction of history to store when forming new address."
	# 	    "(Both wx and fl algorighms)", "type":float}, "flag":"hf", "required_init":True, "default":0.5},
	# 	{ "name":"xor_fraction", "kw":{"help":"Fraction of bits used for xor component in wx algorithm","type":float},
	# 	  "flag":"xf", "required_init":True, "default":0.5},

	def __init__(self, env):
		self.env = env
		self.initialize()

	def initialize(self):
		ma = self.env.pvals["merge_algorithm"]
		self.pvals = {}
		if ma == "wx":
			self.init_wx()
		elif ma == "fl":
			self.init_fl()
		elif ma not in {"hh", "hh2", "wx2"}:
			sys.exit("Invalid merge_algorithm: %s" % ma)

	def init_wx(self):
		# wx Weighted and/or XOR algorithm: created address as two parts: weighted history followed by xor
		# number bits in xor part is xf*N (xf is xor_fraction)
		# weighted history part formed by selecting (1-hf) N bits from item and hf*N bits from history.
		# hf is fraction of history to store when forming new address
		# this function builds the maps selecting bits for the history (not the xor component)
		xf = self.env.pvals["xor_fraction"]
		word_length = self.env.pvals["word_length"]
		xr_len = int(word_length * xf)
		wh_len = word_length - xr_len
		self.pvals["wx"] = {"wh_len":wh_len, "xr_len":xr_len}
		if wh_len > 0:
			# compute weighted history component
			# has two parts, new item bits, then history bits
			hf = self.env.pvals["history_fraction"] # Fraction of history to store when forming new address
			hist_len = int(hf * wh_len)
			assert hist_len > 0, "Must have hist_len > 0, hf (%s) or wh_len (%s) is too small" % (hf, wh_len)
			hist_stride = int(word_length / hist_len)
			start_bit = self.env.pvals["start_bit"]
			hist_bits = [i for i in range(start_bit, word_length, hist_stride)]
			if(len(hist_bits) < hist_len):
				hist_bits.append(0)
			assert len(hist_bits) == hist_len, "len(hist_bits) %s != hist_len (%s)" % (len(hist_bits), hist_len)
			item_len = wh_len - hist_len
			assert item_len > 0, "must have item_len >0, hf (%s) is too large"
			self.pvals["wx"].update( {"hist_bits": hist_bits, "item_len":item_len} )

		# if wh_len > 0:
		# 	# compute weighted history component
		# 	# has two parts, new item bits, then history bits
		# 	hf = self.env.pvals["history_fraction"] # Fraction of history to store when forming new address
		# 	hist_len = int(hf * wh_len)
		# 	assert hist_len > 0, "Must have hist_len > 0, hf (%s) or wh_len (%s) is too small" % (hf, wh_len)
		# 	hist_stride = int(word_length / hist_len)
		# 	start_bit = self.env.pvals["start_bit"]
		# 	hist_bits = [i for i in range(start_bit, word_length, hist_stride)]
		# 	if(len(hist_bits) < hist_len):
		# 		hist_bits.append(0)
		# 	assert len(hist_bits) == hist_len, "len(hist_bits) %s != hist_len (%s)" % (len(hist_bits), hist_len)
		# 	item_len = wh_len - hist_len
		# 	assert item_len > 0, "must have item_len >0, hf (%s) is too large"
		# 	self.pvals["wx"].update( {"hist_bits": hist_bits, "item_len":item_len} )


	def merge_wx(self, item, history):
		# form address as two components.  First (1-xf*N) bits are weighted history. Remaining bits (xf*N)are permuted XOR.
		if(self.pvals["wx"]["wh_len"] > 0):
			# select bits specified by hist_bits and first parte of item
			hist_part = np.concatenate((item[0:self.pvals["wx"]["item_len"]], history[self.pvals["wx"]["hist_bits"]]))
			if self.pvals["wx"]["xr_len"] == 0:
				# no xor component
				assert len(hist_part) == self.env.pvals["word_length"], ("wx algorithm, no xor part, but len(hist_part) %s"
					" does not match word_length (%s)" % (len(hist_part), self.env.pvals["word_length"]))
				return hist_part
		# compute XOR part.  Is permute ( xor component from history) XOR second bits from item.
		hist_input = history[self.pvals["wx"]["wh_len"]:]
		item_input = item[self.pvals["wx"]["wh_len"]:]
		assert len(hist_input) == len(item_input)
		xor_part = np.bitwise_xor(np.roll(hist_input, 1), item_input)
		address = np.concatenate((hist_part, xor_part)) if self.pvals["wx"]["wh_len"] > 0 else xor_part
		assert len(address) == self.env.pvals["word_length"], "wx algorithm, len(address) (%s) != word_length (%s)" % (
			len(address), self.env.pvals["word_length"])
		return address

	def merge_wx2(self, item, history):
		# similar as merge_wx with xf == .5 and hf == .5
		# input history expected to be:  [ hist_part | xor_part ], each part 1/2 word length
		# makes address by new hist_part = .25 item + .5 hist_part, xor_part = (xor_part > 1) XOR .5 item
		start_bit = self.env.pvals["start_bit"]  # is zero or 1
		word_length = self.env.pvals["word_length"]
		wl_half = int(word_length / 2)
		item_half = item[start_bit::2]
		item_qter = item[start_bit::4]
		hist_part = history[start_bit:wl_half]
		hist_half = hist_part[start_bit::2]
		xor_part = history[wl_half:]
		new_xor_part = np.bitwise_xor(np.roll(xor_part, 1), item_half)
		address = np.concatenate((item_qter, hist_half, new_xor_part))
		assert len(address) == word_length, "wx2 algorithm, len(address) (%s) != word_length (%s)" % (
			len(address), word_length)
		return address

	def init_fl(self):
		# From Pentti: The first aN bits should be copied from the first aN bits of
		# the present vector, and the last bN bits should be copied
		# from the LAST bN bits of the permuted history vector.
		hf = self.env.pvals["history_fraction"] # Fraction of history to store when forming new address
		hist_len = int(hf * self.env.pvals["word_length"])
		assert hist_len > 0, "fl algorithm: must have hist_len(%s) > 0, hf (%s) is too small" % (hist_len, hf)
		item_len = self.env.pvals["word_length"] - hist_len
		assert item_len > 0, "fl algorithm: item_len must be > 0, is %s" % item_len
		self.pvals["fl"] = {"hist_len":hist_len, "item_len":item_len}
		# create permutation map for history vector (Probably not needed)
		# shuffled_indices = np.arange(self.env.pvals["word_length"])
		# np.random.shuffle(shuffled_indices)
		# if shuffled_indices[0] == 0:
		# 	# don't let first index be zero, swap it with 2nd index
		# 	# swap from: https://stackoverflow.com/questions/22847410/swap-two-values-in-a-numpy-array
		# 	shuffled_indices[[0,1]] = shuffled_indices[[1,0]]
		# index_map = np.full(self.env.pvals["word_length"], -1, dtype=np.int)
		# current_index = 0
		# for i in range(self.env.pvals["word_length"]):
		# 	index_map[current_index] = shuffled_indices[i]
		# 	current_index = shuffled_indices[i]
		# assert len(np.where(index_map==-1)[0]) == 0, "Did not fill all values in index_map"


	def merge_fl(self, item, history):
		part_1 = item[0:self.pvals["fl"]["item_len"]]
		# np.random.RandomState(seed=42).permutation(history)
		part_2 = np.random.RandomState(seed=42).permutation(history)[-self.pvals["fl"]["hist_len"]:]
		address = np.concatenate((part_1, part_2))
		assert len(address) == self.env.pvals["word_length"], "fl algorithm, len(address) (%s) != word_length (%s)" % (
			len(address), self.env.pvals["word_length"])
		return address

	def merge_hh(self, item, history):
		# original merge, select every other bit from each and concatinate
		# should be same as wx with xor_fraction == 0 and history_fraction = 0.5
		address = np.concatenate((item[0::2],history[0::2]))
		return address

	def merge_hh2(self, item, history):
		# should be more exact match to wx with xor_fraction == 0 and history_fraction = 0.5
		start_bit = self.env.pvals["start_bit"]
		word_length = self.env.pvals["word_length"]
		address = np.concatenate((item[0:int(word_length/2)], history[start_bit::2]))
		assert len(address) == word_length
		return address


	def merge(self, item, history):
		# merge item and history to form new address using the current algorithm
		word_length = self.env.pvals["word_length"]
		assert word_length == len(item) and word_length == len(history)
		ma = self.env.pvals["merge_algorithm"]
		if ma == "wx":
			address = self.merge_wx(item, history)
		elif ma == "wx2":
			address = self.merge_wx2(item, history)
		elif ma == "fl":
			address = self.merge_fl(item, history)
		elif ma == "hh":
			address = self.merge_hh(item, history)
		elif ma == "hh2":
			address = self.merge_hh2(item, history)
		else:
			sys.exit("Invalid merge_algorithm: %s" % ma)
		if self.env.pvals["debug"]:
			print("merge %s, hf=%s, xf=%s, start_bit=%s" % (ma, self.env.pvals["history_fraction"],
				self.env.pvals["xor_fraction"], self.env.pvals["start_bit"]))
			print(" item=%s\n hist=%s\n addr=%s" % (bina2str(item), bina2str(history), bina2str(address)))
		return address

def merge_old(bc, b_hist, history_fraction=0.5, permute=False):
	# bc - binary code for character, b_hist - history binary, hf - history fraction
	# merge binary values bc, b_hist by taking (1 - hf) bits from bc, and hf bits from b_hist then concationate
	# calculate number bits from each
	word_length = len(bc)
	assert word_length == len(b_hist)
	n_bits_hist = int(word_length * history_fraction)
	n_bits_c = word_length - n_bits_hist
	hist_larger = n_bits_hist 
	stride 
	if permute:
			b3 = np.roll(np.concatenate((b1[0::2],np.roll(b2[0::2], 1))), 1)
	else:
		b3 = np.concatenate((b1[0::2],b2[0::2]))
	# if permute:
	# 	b3 = np.roll(b3, 1)
	return b3

class Char_map:
	# maps each character in a sequence to a random binary word
	# below specifies maximum number of unique characters
	max_unique_chars = 50

	def __init__(self, seq, word_length = 128):
		# seq - a Sequence object
		# word_length - number of bits in random binary word
		assert word_length % 2 == 0, "word_length must be even"
		self.word_length = word_length
		self.chars = ''.join(sorted(list(set(list(''.join(seq))))))  # sorted string of characters appearing in seq
		self.chars += "#"  # stop char - used to indicate end of sequence
		self.check_length()
		self.binary_vals = initialize_binary_matrix(self.max_unique_chars, word_length)
	
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
		return self.binary_vals[index]

	def bin2char(self, b, nret = 3):
		# returns array of top matching characters and the hamming distance (# of bits not matching)
		# b is a binary array, same length as word_length
		# nret is the number of matches to return
		assert self.word_length == len(b)
		top_matches = find_matches(self.binary_vals[0:len(self.chars),:], b, nret)
		top_matches = [ (self.chars[x[0]], x[1]) for x in top_matches ]
		return top_matches


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

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		d = data.copy()
		d[d==0] = -1  # replace zeros in data with -1
		for i in top_matches:
			self.data_array[i] += d
		if self.debug:
			print("store\n addr=%s\n data=%s" % (bina2str(address), bina2str(data)))

	def read(self, address):
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		i = top_matches[0]
		sum = self.data_array[i].copy()
		for i in top_matches[1:]:
			sum += self.data_array[i]
		sum[sum<1] = 0   # replace values less than 1 with zero
		sum[sum>0] = 1   # replace values greater than 0 with 1
		if self.debug:
			print("read\n addr=%s\n data=%s" % (bina2str(address), bina2str(sum)))
		return sum

	def clear(self):
		# set data_array contents to zero
		self.data_array.fill(0)

def bina2str(address):
	# convert binary array address to a string
	binary_string = "".join(["%s" % i for i in address])
	wlength = 8  # insert space between each 8 characters.  From: 
	# https://stackoverflow.com/questions/10070434/how-do-i-insert-a-space-after-a-certain-amount-of-characters-in-a-string-using-p
	binary_string_with_spaces = ' '.join(binary_string[i:i+wlength] for i in range(0,len(binary_string),wlength))
	return binary_string_with_spaces

def expand(b):
	# make array b two times as long by duplicating every element
	# from: https://stackoverflow.com/questions/17619415/how-do-i-combine-two-numpy-arrays-element-wise-in-python
	b2 = np.empty((b.shape[0]*2), dtype=b.dtype)
	b2[0::2] = b
	b2[1::2] = b
	return b2

def hamming(b1, b2):
	# compute hamming distance
	ndiff = np.count_nonzero(b1!=b2)
	return ndiff

def recall(prefix, env):
	# recall sequence starting with prefix
	# build address to start searching
	word_length = env.pvals["word_length"]
	wl_half = int(word_length / 2)
	threshold = word_length * env.pvals["char_match_fraction"]
	ma = env.pvals["merge_algorithm"]
	debug = env.pvals["debug"]
	# start_bit = env.pvals["start_bit"]
	b0 = env.cmap.char2bin(prefix[0])  # binary word associated with first character
	# if ma == "wx2":
	# 	# for wx2 algorithm, initial history is .5 b0 + 0.5 b0; data to store is 0.5 char_2 + 0.5 b0
	# 	b0_half = b0[start_bit::2]
	# 	init_hist = np.np.concatenate((b0_half, b0_half))
	# 	address = env.merge.merge(b0, init_hist)
	# else:
	# 	address = env.merge.merge(b0, b0)
	address = env.merge.merge(b0, b0)
	for char in prefix[1:]:
		b = env.cmap.char2bin(char)
		address = env.merge.merge(b, address)
	found = [ prefix, ]
	word2 = prefix
	if ma  == "wx2":
		computed_address = address
	# now read sequence using prefix address
	ccount = 0
	while True:
		b = env.sdm.read(address)
		if ma == "wx2":
			bchar = expand(b[0:wl_half])
			xor_part = b[wl_half:]
			if True: # debug:
				computed_xor_part = computed_address[wl_half:]
				print(" xor hamming =%s" % hamming(computed_xor_part, xor_part))
				# print(" xor hamming =%s, computed_part = %s, found=%s" % (
				# 	hamming(computed_xor_part, xor_part), bina2str(computed_xor_part), bina2str(xor_part)))
		else:
			bchar = b
		top_matches = env.cmap.bin2char(bchar)
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
			bchar = env.cmap.char2bin(found_char)
		if ma == "wx2":
			history = np.concatenate((address[0:wl_half], xor_part))
			if debug:
				print("computing address")
			computed_address = env.merge.merge(bchar, computed_address)
		else:
			history = address
		new_address = env.merge.merge(bchar, history)
		diff = np.count_nonzero(history!=new_address)
		found[-1].append("addr_diff=%s" % diff)
		if diff == 0:
			found[-1].append("\nfound fixed point, address=\n%s" % (
				bina2str(new_address)))
			break
		address = new_address
		word2 += found_char
		ccount += 1
		if ccount > 50:
			break
	return [found, word2]

def store_strings(env, strings):
	ma = env.pvals["merge_algorithm"]
	wl_half = int(env.pvals["word_length"]/2)
	start_bit = env.pvals["start_bit"]
	debug = env.pvals["debug"] == 1
	for string in strings:
		print("storing '%s'" % string)
		b0 = env.cmap.char2bin(string[0])  # binary word associated with first character
		address = env.merge.merge(b0, b0)
		for char in string[1:]+"#":  # add stop character at end
			bchar = env.cmap.char2bin(char)
			if ma == "wx2":
				if debug:
					print(" bchr:%s - %s" % (bina2str(bchar), char))
				# store .5 bchar + end of address (xor part)
				value = np.concatenate((bchar[start_bit::2], address[wl_half:]))
			else:
				value = bchar
			env.sdm.store(address, value)
			address = env.merge.merge(bchar, address)
		env.record_saved_string(string)

def test_merge_convergence(env):
	# test how fast address for substring converges to address for full string
	string = "abcdefghijklmnopqrstuvwxyz"
	b0 = env.cmap.char2bin(string[0])
	address = env.merge.merge(b0, b0)
	b1 = env.cmap.char2bin(string[1])
	address = env.merge.merge(b1, address)
	subaddr = env.merge.merge(b1, b1)
	result = []
	for char in string[2:]:
		bchar = env.cmap.char2bin(char)
		address = env.merge.merge(bchar, address)
		subaddr = env.merge.merge(bchar, subaddr)
		distance = hamming(address, subaddr)
		result.append((char, distance))
	print("Convergence for merge %s, hf=%s, xf=%s, start_bit=%s:\n%s" % (env.pvals["merge_algorithm"],
		env.pvals["history_fraction"], env.pvals["xor_fraction"], env.pvals["start_bit"], result))


def store_param_strings(env):
	# store strings provided as parameters in command line (or updated)
	store_strings(env, env.pvals["string_to_store"])

def recall_strings(env, strings, prefix_mode = False):
	# recall list of strings starting with first character of each
	# if prefix_mode is True, use entire string as prefix for recall
	error_count = 0
	for word in strings:
		print ("\nRecall '%s'" % word)
		prefix = word if prefix_mode else word[0]
		found, word2 = recall(prefix, env)
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
		print("%s words, %s errors" % (len(strings), error_count))

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
