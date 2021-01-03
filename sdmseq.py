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
			" matching character to item memory","type":float},"flag":"cmf", "required_init":"", "default":0.25},
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
	 	  "flag":"xf", "required_init":"m", "default":0.5},
	 	{ "name":"first_bin_fraction", "kw":{"help":"First bin fraction, fraction of bits of item used for first bin "
	 	  "in wx algorithm OR an integer > 1 giving number of equal history bins OR an integer + 0.5 to store "
	 	  "reverse character in value (wx2 algorithm)", "type":float}, "flag":"fbf",
	 	  "required_init":"m", "default":8},
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
		# repeat = self.pvals["merge_algorithm"] == "wx2"  # if wx2 algorithm, have first half of item words match second half
		repeat = False
		print("Initializing.")
		self.cmap = Char_map(self.pvals["string_to_store"], word_length=word_length, repeat=repeat)
		self.sdm = Sdm(address_length=word_length, word_length=word_length, num_rows=self.pvals["num_rows"],
			nact = self.pvals["activation_count"], debug=self.pvals["debug"])
		self.ma = getattr(sys.modules[__name__], "Ma_" + self.pvals["merge_algorithm"])(self)
		# self.merge = Merge(self)
		self.saved_strings = []
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
		elif cmd == "v" and len(arg) > 0:
			print("reverse recall prefix %s" % arg)
			reverse_recall_strings(env, shlex.split(arg))
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
	# m is 2-d array of binary values, first dimension if value index, second is binary number
	# b is binary number to match
	# nret is number of top matches to return
	# returns sorted tuple (i, c) where i is index and c is number of non-matching bits (0 is perfect match)
	# if index_only is True, only return the indices, not the c
	# if match_bits is not None, it is the number of bits to match (subset), otherwise full length is used
	assert len(m.shape) == 2, "array to match must be 2-d"
	# assert m.shape[1] == len(b), "array element size does not match size of match binary value"
	if match_bits is None:
		match_bits = len(b)
	assert match_bits <= len(b), "match_bits for find_matchs too long (%s), must be less than (%s)" % (match_bits, len(b))
	matches = []
	for i in range(m.shape[0]):
		ndiff = np.count_nonzero(m[i][0:match_bits]!=b[0:match_bits])
		matches.append( (i, ndiff) )
	matches.sort(key=itemgetter(1))
	# if ties in hamming distance, make sure the first rows (lowest index number) are selected
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

def initialize_binary_matrix(nrows, ncols, repeat=False):
	# create binary matrix with each row having binary random number
	assert ncols % 2 == 0, "ncols must be even"
	bm = np.random.randint(2, size=(nrows, ncols), dtype=np.int8)
	if repeat:
		# make second half of each word same as first half.  Used in wx2 method.
		bm[:,int(ncols/2):] = bm[:,0:int(ncols/2)]
	# if wanted same number of zeros and ones, code below could be used.
	# hw = ncols / 2
	# bm = np.zeros( (nrows, ncols), dtype=dtype=np.int8) # bm - binary_matrix
	# for i in range(nrows):
	# 	bm[i][0:hw] = 1				# set half of row to 1, then shuffle
	# 	np.random.shuffle(bm[i])
	return bm

def make_permutation_map(length):
	# create permutation map for history vector.  This done because the numpy function I found
	# np.random.RandomState(seed=42).permutation(b) does not always include all the elements in
	# the permutation
	shuffled_indices = np.arange(length)
	np.random.shuffle(shuffled_indices)
	# map each index to the next one in the shuffled list
	map_index = [shuffled_indices[(i+1) % length] for i in range(length)]
	return map_index

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

	def add(self, address, value, char):
		# add character char to sequence
		# address and value is current address and value at that address before adding char to sequence
		# Returns (new_address, new_value) - new address and the value to store at that address
		bchar2 = self.env.cmap.char2bin(char)
		if address is None:
			# if this is the start of the sequence, use binary for char for both address and value
			assert value is None
			address = bchar2
			value = bchar2
		bchar1 = self.get_bchar1(value)
		new_address = self.make_new_address(address, bchar1)
		new_value = self.make_new_value(address, new_address, bchar2)
		return (new_address, new_value)

	def next(self, address, value):
		# get next address and character in sequence
		# given address and value at that address, return [new_address, char, top_matches] character in sequence
		bchar1_part = self.get_bchar1_part(value)
		top_matches = self.env.cmap.bin2char(bchar1_part)
		char = top_matches[0][0]
		if char == "#" or top_matches[0][1] > self.pvals["char_match_threshold"]:
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
		bchar0_part = self.get_bchar0_part(address)
		top_matches = self.env.cmap.bin2char(bchar0_part)
		char = top_matches[0][0]
		if top_matches[0][1] > self.pvals["char_match_threshold"]:
			# found stop char, or hamming distance to top match is over threshold
			new_address = None
		else:
			if self.reverse_end_found(address, value, char):
				new_address = None
			else:
				new_address = self.make_reverse_address(address, value, char, top_matches)
		return [new_address, char, top_matches]

	# following default functions MAY be overridden in subclassses

	def initialize(self):
		print("Initialize merge_algorithm '%s'" % self.name)

	def get_bchar1_part(self, value):
		# return bits of bchar1 in value (but don't expand to full length if not already full length)
		return value

	def get_bchar1(self, value):
		# return full length
		return value

	def make_new_value(self, address, new_address, bchar2):
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
			# include reverse character (for wx2 algorithm), indicated by fbf being integer>1 + .5
			include_reverse_char = self.name == "wx2" and fbf > 1.0 and ((round(fbf * 10.0) % 10) == 5)
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
			print("wx2 init:")
			print("item_len=%s" % bin0_len)
			print("bin_sizes=%s" % bin_sizes)
			print("hist_bits=%s" % ind)
			self.pvals.update( {"hist_bits": ind, "item_len":bin0_len, "bin_sizes":bin_sizes})
			if include_reverse_char:
				# used for wx2 if allow reverse recall
				last_char_position = sum(bin_sizes[0:-1])  # position of reverse character
				len_item_part = round(bin0_len / 2.0)
				len_reverse_part = bin0_len - len_item_part
				self.pvals.update( {"last_char_position":last_char_position, "len_item_part":len_item_part,
					"len_reverse_part":len_reverse_part} )
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
	# 	assert self.name in ("wx", "wx2")
	# 	item_tp2 = self.env.cmap.char2bin(char)
	# 	# get bits for <item_t+1> from value
	# 	if self.name == "wx2":
	# 		item_tp1_len = self.pvals["len_item_part"] if self.pvals["include_reverse_char"] else self.pvals["item_len"]
	# 		item_tp1 = self.env.cmap.part2full(value[0:item_tp1_len])
	# 	else:
	# 		item_tp1 = value
	# 	# form address as two components.  First (1-xf*N) bits are weighted history. Remaining bits (xf*N)are permuted XOR.
	# 	if(self.pvals["wh_len"] > 0):
	# 		# select bits specified by hist_bits and first part of item
	# 		hist_part = np.concatenate((item_tp1[0:self.pvals["item_len"]], address[self.pvals["hist_bits"]]))
	# 		if self.pvals["xr_len"] == 0:
	# 			# no xor component
	# 			assert len(hist_part) == self.env.pvals["word_length"], ("wx algorithm, no xor part, but len(hist_part) %s"
	# 				" does not match word_length (%s)" % (len(hist_part), self.env.pvals["word_length"]))
	# 			return (hist_part, item_tp2)
	# 	# compute XOR part.  Is permute ( xor component from address) XOR leading bits from <item_t+1>.
	# 	# if wh_len is zero then is pure XOR algorithm
	# 	addr_input = address[self.pvals["wh_len"]:]
	# 	item_input = item_tp1[0:self.pvals["xr_len"]]
	# 	assert len(addr_input) == len(item_input)
	# 	xor_part = np.bitwise_xor(np.roll(addr_input, 1), item_input)
	# 	new_address = np.concatenate((hist_part, xor_part)) if self.pvals["wh_len"] > 0 else xor_part
	# 	assert len(new_address) == self.env.pvals["word_length"], "wx algorithm, len(new_address) (%s) != word_length (%s)" % (
	# 		len(new_address), self.env.pvals["word_length"])
	# 	# now get new_value
	# 	if self.name == "wx2":
	# 		if self.pvals["include_reverse_char"]:
	# 			item_input = np.concatenate((item_tp2[0:self.pvals["len_item_part"]], 
	# 				address[self.pvals["last_char_position"]:self.pvals["wh_len"]]))
	# 			assert len(item_input) == self.pvals["item_len"]
	# 		else:
	# 			item_input = item_tp2[0:self.pvals["item_len"]]
	# 		new_value = np.concatenate((item_input, new_address[self.pvals["item_len"]:]))
	# 	else:
	# 		new_value = item_tp2
	# 	assert len(new_value) == env.pvals["word_length"]
	# 	return (new_address, new_value)
		# if debug:
		# 	print(" bchr:%s - %s" % (bina2str(bchar), char))


	def make_new_address(self, address, bchar1):
		# form address as two components.  First (1-xf*N) bits are weighted history. Remaining bits (xf*N)are permuted XOR.
		if(self.pvals["wh_len"] > 0):
			# select bits specified by hist_bits and first part of item
			hist_part = np.concatenate((bchar1[0:self.pvals["item_len"]], address[self.pvals["hist_bits"]]))
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


	# def next(self, address, value):
	# 	# get next address and character in sequence
	# 	# given address and value at that address, return [new_address, char, top_matches]
	# 	# where new_address is the next address and char is the character at the passed in address & value
	# 	# top matches are a list of the three top matching characters
	# 	assert self.name == "wx"
	# 	top_matches = self.env.cmap.bin2char(value)
	# 	char = top_matches[0][0]
	# 	if char == "#" or top_matches[0][1] > self.pvals["char_match_threshold"]
	# 		# found stop char, or hamming distance to top match is over threshold
	# 		new_address = None
	# 	else:
	# 		new_address = make_new_address(address, )
	# 		[new_address, value] = self.add(char, address)
	# 	return [new_address, char, top_matches]

class Ma_wx2(Ma_wx):
	# similar to wx, except data value includes xor part

	def get_bchar1_part(self, value):
		# get bits for <char_t+1> from value
		assert self.name == "wx2"
		bc1p_len = self.pvals["len_item_part"] if self.pvals["include_reverse_char"] else self.pvals["item_len"]
		return value[0:bc1p_len]

	def get_bchar1(self, value):
		bchar1_part = self.get_bchar1_part(value)
		bchar1 = self.env.cmap.part2full(bchar1_part)
		return bchar1

	def make_new_value(self, address, new_address, bchar2):
		assert self.name == "wx2"
		if self.pvals["include_reverse_char"]:
			item_input = np.concatenate((bchar2[0:self.pvals["len_item_part"]], 
				address[self.pvals["last_char_position"]:self.pvals["wh_len"]]))
			assert len(item_input) == self.pvals["item_len"]
		else:
			item_input = bchar2[0:self.pvals["item_len"]]
		new_value = np.concatenate((item_input, new_address[self.pvals["item_len"]:]))
		assert len(new_value) == self.env.pvals["word_length"]
		return new_value

	def get_bchar0_part(self, address):
		# return bits of item_t0 (first bin of address in wx2 algorithm)
		return address[0:self.pvals["item_len"]]

	def reverse_end_found(self, address, value, char):
		# return True if bins after first (in address) plus reverse part (in value) encode for char
		tail_bins = np.concatenate((address[self.pvals["item_len"]:self.pvals["wh_len"]],
			value[self.pvals["len_item_part"]:self.pvals["item_len"]]))
		top_matches = self.env.cmap.bin2char(tail_bins)
		char2 = top_matches[0][0]
		hamming = top_matches[0][1]
		return char == char2 and hamming < self.pvals["char_match_threshold"]

	def make_reverse_address(self, address, value, char, top_matches):
		# if cannot find character matching reverse char, include that in "top_matches"
		bchar0 = self.env.cmap.char2bin(char)
		rev_part = value(value[self.pvals["len_item_part"]:self.pvals["item_len"]])
		top_matches2 = self.env.cmap.bin2char(tail_bins)
		rev_char, hamming = top_matches[0]
		if hamming > self.pvals["char_match_threshold"] * (len(rev_part) / env.pvals["word_length"]):
			top.matches.append("rev_char lookup failed: %s" % top_matches2)
			new_address = None
		else:
			brev_char = self.env.cmap.char2bin(rev_char)
			new_xor_part = np.roll(np.bitwise_xor(address[self.pvals["wh_len"]:],bchar0[0:self.pvals["xr_len"]]),-1)
			new_address = np.concatenate((address[self.pvals["item_len"]:self.pvals["wh_len"]],
				brev_char[0:self.pvals["item_len"]], new_xor_part))
		return new_address


	# def prev(self, address, value):
	# 	# get previous address and character in sequence (reverse recall)
	# 	# given address and value at that address, return [new_address, char] character in reverse sequence
	# 	# if at beginning of sequence, (char is first char) return new_address == None 
	# 	assert self.pvals["include_reverse_char"]
	# 	bchar_t_part = address[0:self.pvals["item_len"]]
	# 	top_matches = self.env.cmap.bin2char(bchar_t_part)
	# 	if top_matches[0][1] > self.pvals["char_match_threshold"]:
	# 	bcrev_part = value[self.pvals["len_item_part"]:self.pvals["item_len"]]
	# 			top_matches = self.env.cmap.bin2char(bchar1_part)
	# 	char = top_matches[0][0]
	# 	if char == "#" or top_matches[0][1] > self.pvals["char_match_threshold"]
	# 		# found stop char, or hamming distance to top match is over threshold
	# 		new_address = None
	# 	else:
	# 		if top_matches[0][1] > 0 or len(bchar1_part) < self.env.pvals["word_length"]:
	# 			# get cleaned up binary for bchar1
	# 			bchar1_part = self.env.cmap.char2bin(char)
	# 		new_address = make_new_address(address, bchar1_part)
	# 	return [new_address, char, top_matches]


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
		self.pvals["fl"] = {"hist_len":hist_len, "item_len":item_len}
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
		if ma in ("wx", "wx2"):
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
			# this code has a bug, may not work unless history_fraction == 0.5
			fbf = self.env.pvals["first_bin_fraction"]  # fraction of wh_len used for first bin (bits from current item)
			if fbf > 1.0:
				# equal size bins
				num_bins = int(fbf)
				bits_per_bin = int(wh_len/num_bins)
				remaining_bits = wh_len % num_bins
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
			offset = bin0_len
			offset = 0
			for i in range(1,len(bin_sizes)):
				bin_size = bin_sizes[i]
				for j in range(bin_size):
					ind.append(j+offset)
				offset += bin_sizes[i-1]
			assert len(ind) + bin0_len + xr_len == word_length, ("wx hist init mismatch, ind=%s, bin0_len=%s, xr_len=%s" %
				(ind, bin0_len, xr_len))
			print("wx2 init:")
			print("item_len=%s" % bin0_len)
			print("bin_sizes=%s" % bin_sizes)
			print("hist_bits=%s" % ind)
			self.pvals["wx"].update( {"hist_bits": ind, "item_len":bin0_len, "bin_sizes":bin_sizes} )


			# hf = self.env.pvals["history_fraction"] # Fraction of history to store when forming new address
			# hist_len = int(hf * wh_len)
			# assert hist_len > 0, "Must have hist_len > 0, hf (%s) or wh_len (%s) is too small" % (hf, wh_len)
			# hist_stride = int(word_length / hist_len)
			# start_bit = self.env.pvals["start_bit"]
			# hist_bits = [i for i in range(start_bit, word_length, hist_stride)]
			# if(len(hist_bits) < hist_len):
			# 	hist_bits.append(0)
			# assert len(hist_bits) == hist_len, "len(hist_bits) %s != hist_len (%s)" % (len(hist_bits), hist_len)
			# item_len = wh_len - hist_len
			# assert item_len > 0, "must have item_len >0, hf (%s) is too large"
			# self.pvals["wx"].update( {"hist_bits": hist_bits, "item_len":item_len} )

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
		# if wh_len is zero then is pure XOR algorithm
		hist_input = history[self.pvals["wx"]["wh_len"]:]
		item_input = item[self.pvals["wx"]["wh_len"]:]
		assert len(hist_input) == len(item_input)
		xor_part = np.bitwise_xor(np.roll(hist_input, 1), item_input)
		address = np.concatenate((hist_part, xor_part)) if self.pvals["wx"]["wh_len"] > 0 else xor_part
		assert len(address) == self.env.pvals["word_length"], "wx algorithm, len(address) (%s) != word_length (%s)" % (
			len(address), self.env.pvals["word_length"])
		return address


	def merge_wx2(self, item, history):
		# similar as merge_wx, but value stored is different
		return self.merge_wx(item, history)



	# def merge_wx2_orig(self, item, history):
	# 	# similar as merge_wx with xf == .5 and hf == .5
	# 	# input history expected to be:  [ hist_part | xor_part ], each part 1/2 word length
	# 	# makes address by new hist_part = .25 item + .5 hist_part, xor_part = (xor_part > 1) XOR .5 item
	# 	start_bit = self.env.pvals["start_bit"]  # is zero or 1
	# 	word_length = self.env.pvals["word_length"]
	# 	wl_half = int(word_length / 2)
	# 	item_half = item[0:wl_half]  # start_bit::2]
	# 	item_qter = item[start_bit::4]
	# 	hist_part = history[start_bit:wl_half]
	# 	hist_half = hist_part[start_bit::2]
	# 	xor_part = history[wl_half:]
	# 	new_xor_part = np.bitwise_xor(np.roll(xor_part, 1), item_half)
	# 	address = np.concatenate((item_qter, hist_half, new_xor_part))
	# 	assert len(address) == word_length, "wx2 algorithm, len(address) (%s) != word_length (%s)" % (
	# 		len(address), word_length)
	# 	return address

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
		self.pvals["shuffle_map"] = make_permutation_map(self.env.pvals["word_length"])

	def merge_fl(self, item, history):
		part_1 = item[0:self.pvals["fl"]["item_len"]]
		# np.random.RandomState(seed=42).permutation(history)
		# routine "np.random.RandomState(seed=42).permutation(history)" skips some elements, can't use following
		# part_2 = np.random.RandomState(seed=42).permutation(history)[-self.pvals["fl"]["hist_len"]:]
		part_2 = history[self.pvals["shuffle_map"]][-self.pvals["fl"]["hist_len"]:]
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

	def __init__(self, seq, word_length = 128, repeat=False):
		# seq - a Sequence object
		# word_length - number of bits in random binary word
		# repeat - make second half of each word same as first half (used for wx2)
		assert word_length % 2 == 0, "word_length must be even"
		self.word_length = word_length
		self.chars = ''.join(sorted(list(set(list(''.join(seq))))))  # sorted string of characters appearing in seq
		self.chars += "#"  # stop char - used to indicate end of sequence
		self.check_length()
		self.repeat = True
		self.binary_vals = initialize_binary_matrix(self.max_unique_chars, word_length, repeat)
	
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

	def store(self, address, data):
		# store binary word data at top nact addresses matching address
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True)
		d = data.copy()
		d[d==0] = -1  # replace zeros in data with -1
		for i in top_matches:
			self.data_array[i] += d
		if self.debug:
			print("store\n addr=%s\n data=%s" % (bina2str(address), bina2str(data)))

	def read(self, address, match_bits=None):
		top_matches = find_matches(self.addresses, address, self.nact, index_only = True, match_bits=match_bits)
		i = top_matches[0]
		sum = self.data_array[i].copy()
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
	binary_string = "".join(["%s" % i for i in address])
	# following from: https://stackoverflow.com/questions/2072351/python-conversion-from-binary-string-to-hexadecimal
	hex_string = '%0*x' % ((len(binary_string) + 3) // 4, int(binary_string, 2))
	wlength = 8  # insert space between each 8 characters.  From: 
	# https://stackoverflow.com/questions/10070434/how-do-i-insert-a-space-after-a-certain-amount-of-characters-in-a-string-using-p
	# binary_string_with_spaces = ' '.join(binary_string[i:i+wlength] for i in range(0,len(binary_string),wlength))
	hex_string_with_spaces = ' '.join(hex_string[i:i+wlength] for i in range(0,len(hex_string),wlength))
	return hex_string_with_spaces

# def expand(b):
# 	# make array b two times as long by duplicating every element
# 	# from: https://stackoverflow.com/questions/17619415/how-do-i-combine-two-numpy-arrays-element-wise-in-python
# 	b2 = np.empty((b.shape[0]*2), dtype=b.dtype)
# 	b2[0::2] = b
# 	b2[1::2] = b
# 	return b2

def hamming(b1, b2):
	# compute hamming distance
	ndiff = np.count_nonzero(b1!=b2)
	return ndiff


def recall(prefix, env):
	# recall sequence starting with prefix
	# build address to start searching
	word_length = env.pvals["word_length"]
	# threshold = word_length * env.pvals["char_match_fraction"]
	max_num_recalled_chars = len(max(env.saved_strings, key=len)) + 5
	ma = env.pvals["merge_algorithm"]
	if ma == "wx2":
		wh_len = env.ma.pvals["wh_len"]
		item_len = env.ma.pvals["item_len"]
	debug = env.pvals["debug"]
	address = None
	value = None
	for char in prefix:
		address, value = env.ma.add(address, value, char)
	found = [ prefix[0:-1], ]
	word2 = prefix[0:-1]
	if ma  == "wx2":
		if len(prefix) > 1:
			# iteratively read value at address until converges (no decrease in hamming distance between xor part in
			# address and value read)
			pb_len = sum(env.ma.pvals["bin_sizes"][0:len(prefix)]) # number prefix bits
			prev_address = address
			hconverge=[]
			max_steps = 10
			while True:
				found_value = env.sdm.read(address)
				address = np.concatenate((address[0:pb_len], found_value[pb_len:]))
				hdiff = hamming(prev_address, address)
				hconverge.append(hdiff)
				if hdiff == 0 or len(hconverge) > max_steps:
					break
				prev_address = address
			f_non_zero = np.count_nonzero(found_value) / len(found_value)
			print("Iterative converge in %s steps, f_non_zero=%s, hdiff=%s" % (len(hconverge), f_non_zero, hconverge))
			if f_non_zero < 0.1:
				print("Failed to converge to valid data (found_value near zero)")
				return [found, word2]
			if debug:
				print(" found_value = %s" % bina2str(found_value))
				print(" created address used to start reading sequence:\n %s" % bina2str(address))
	computed_address = address
	computed_value = value
	# now read sequence using address derived from prefix
	ccount = 0
	while True:
		value = env.sdm.read(address)
		if ma == "wx2":
			if debug:
				print(" reading, addr=%s, value=%s" % (bina2str(address), bina2str(value)))
			# value read (b) has char part followed by xor part
			recalled_xor_part = value[wh_len:]
			computed_xor_part = computed_address[wh_len:]
			xor_hamming = " xor hamming =%s" % hamming(computed_xor_part, recalled_xor_part)
			# following to use computed_xor_part for all subsequent recall, ignoring what is in sdm for xor part
			# computed_xor_part = xor_part
			match_bits = item_len
			# test replacing xor part of address with read xor part:
			# address = np.concatenate((address[0:wh_len], recalled_xor_part))
		else:
			# bchar = b
			xor_hamming = None
			match_bits = None
		new_address, found_char, top_matches = env.ma.next(address, value)
		found.append(top_matches)
		if new_address is None:
			# found stop char, or hamming distance to top match is over threshold
			break
		# at this point, have:
		#  address - address of value just read from memory
		#  found_char - found character (printable code) derived from b
		# wx2 method only:
		#  recalled_xor_part - xor part read from memory (from b)
		#  computed_xor_part - xor part computed at each iteration using previous address and value
		new_computed_address, new_computed_value = env.ma.add(computed_address, computed_value, found_char)
		diff = np.count_nonzero(address!=new_address)
		found[-1].append("addr_diff=%s" % diff)
		if xor_hamming:
			found[-1].append(xor_hamming)
		if diff == 0:
			found[-1].append("found fixed point in address")
			    #, address=\n%s" % (bina2str(new_address)))
			break
		address = new_address
		computed_address = new_computed_address
		word2 += found_char
		ccount += 1
		if ccount > max_num_recalled_chars:
			break
	return [found, word2]


def recall_old(prefix, env):
	# recall sequence starting with prefix
	# build address to start searching
	word_length = env.pvals["word_length"]
	threshold = word_length * env.pvals["char_match_fraction"]
	max_num_recalled_chars = len(max(env.saved_strings, key=len)) + 5
	ma = env.pvals["merge_algorithm"]
	if ma == "wx2":
		wh_len = env.merge.pvals["wx"]["wh_len"]
		item_len = env.merge.pvals["wx"]["item_len"]
	debug = env.pvals["debug"]
	b0 = env.cmap.char2bin(prefix[0])  # binary word associated with first character
	address = env.merge.merge(b0, b0)
	for char in prefix[1:]:
		b = env.cmap.char2bin(char)
		address = env.merge.merge(b, address)
	found = [ prefix, ]
	word2 = prefix
	if ma  == "wx2":
		if len(prefix) > 1:
			# iteratively read value at address until converges (no decrease in hamming distance between xor part in
			# address and value read)
			pb_len = sum(env.merge.pvals["wx"]["bin_sizes"][0:len(prefix)]) # number prefix bits
			prev_address = address  # a large number so first found_value reduces hdiff
			hconverge=[]
			max_steps = 10
			while True:
				found_value = env.sdm.read(address)
				address = np.concatenate((address[0:pb_len], found_value[pb_len:]))
				hdiff = hamming(prev_address, address)
				hconverge.append(hdiff)
				if hdiff == 0 or len(hconverge) > max_steps:
					break
				prev_address = address
			f_non_zero = np.count_nonzero(found_value) / len(found_value)
			print("Iterative converge in %s steps, f_non_zero=%s, hdiff=%s" % (len(hconverge), f_non_zero, hconverge))
			if f_non_zero < 0.1:
				print("Failed to converge to valid data (found_value near zero)")
				return [found, word2]
			if debug:
				print(" found_value = %s" % bina2str(found_value))
				print(" created address used to start reading sequence:\n %s" % bina2str(address))
	computed_address = address
	# now read sequence using address derived from prefix
	ccount = 0
	while True:
		b = env.sdm.read(address)
		if ma == "wx2":
			if debug:
				print(" reading, addr=%s, result b=%s" % (bina2str(address), bina2str(b)))
			# value read (b) has char part followed by xor part
			# bchar = np.concatenate((b[0:wl_half], b[0:wl_half]))
			recalled_xor_part = b[wh_len:]
			# computed_xor_part = computed_address[wh_len:]
			computed_xor_part = computed_address[wh_len:]
			xor_hamming = " xor hamming =%s" % hamming(computed_xor_part, recalled_xor_part)
			# following to use computed_xor_part for all subsequent recall, ignoring what is in sdm for xor part
			# computed_xor_part = xor_part
			match_bits = item_len
			# test replacing xor part of address with read xor part:
			# address = np.concatenate((address[0:wh_len], recalled_xor_part))
		else:
			# bchar = b
			xor_hamming = None
			match_bits = None
		top_matches = env.cmap.bin2char(b, match_bits=match_bits)
		found.append(top_matches)
		found_char = top_matches[0][0]
		if found_char == "#" or top_matches[0][1] > threshold:
			# found stop char, or hamming distance to top match is over threshold
			break
		# get full binary word corresponding to character
		if top_matches[0][1] > 0 or match_bits is not None:
			bchar = env.cmap.char2bin(found_char)
		else:
			bchar = b
		# at this point, have:
		#  address - address of value just read from memory
		#  b - value read using "address"
		#  found_char - found character (printable code) derived from b
		#  bchar - binary word corresponding to found_char
		# wx2 method only:
		#  recalled_xor_part - xor part read from memory (from b)
		#  computed_xor_part - xor part computed at each iteration using previous address and bchar
		# Now need to build the new address
		# create history to use in merge to create new address for reading next character in the sequence
		# if ma == "wx2":
		# 	# substitute computed 
		# 	history = np.concatenate((address[0:wh_len], computed_xor_part))
		# 	# if debug:
		# 	# 	print("computing address")
		# 	# computed_address = env.merge.merge(bchar, computed_address)
		# else:
		# 	history = address
		new_computed_address = env.merge.merge(bchar, computed_address)
		new_address = env.merge.merge(bchar, address)
		diff = np.count_nonzero(address!=new_address)
		found[-1].append("addr_diff=%s" % diff)
		if xor_hamming:
			found[-1].append(xor_hamming)
		if diff == 0:
			found[-1].append("found fixed point in address")
			    #, address=\n%s" % (bina2str(new_address)))
			break
		address = new_address
		computed_address = new_computed_address
		word2 += found_char
		ccount += 1
		if ccount > max_num_recalled_chars:
			break
	return [found, word2]

def store_strings(env, strings):
	# debug = env.pvals["debug"] == 1
	for string in strings:
		print("storing '%s'" % string)
		address = None
		value = None
		for char in string+"#":  # add stop character at end
			address, value = env.ma.add(address, value, char)
			env.sdm.store(address, value)
		env.record_saved_string(string)
		# if debug:
		# 	print(" bchr:%s - %s" % (bina2str(bchar), char))


# def store_strings_old3(env, strings):
# 	ma = env.pvals["merge_algorithm"]
# 	if ma == "wx2":
# 		# wh_len = env.merge.pvals["wx"]["wh_len"]
# 		item_len = env.merge.pvals["wx"]["item_len"]
# 	debug = env.pvals["debug"] == 1
# 	for string in strings:
# 		print("storing '%s'" % string)
# 		b0 = env.cmap.char2bin(string[0])  # binary word associated with first character
# 		address = env.merge.merge(b0, b0)
# 		for char in string[1:]+"#":  # add stop character at end
# 			bchar = env.cmap.char2bin(char)
# 			if ma == "wx2":
# 				# store first part of character followed by xor part
# 				value = np.concatenate((bchar[0:item_len], address[item_len:]))
# 				assert len(value) == env.pvals["word_length"]
# 				if debug:
# 					print(" bchr:%s - %s" % (bina2str(bchar), char))
# 			else:
# 				value = bchar
# 			env.sdm.store(address, value)
# 			address = env.merge.merge(bchar, address)
#		env.record_saved_string(string)


# def store_strings_old2(env, strings):
# 	ma = env.pvals["merge_algorithm"]
# 	if ma == "wx2":
# 		wh_len = env.merge.pvals["wx"]["wh_len"]
# 	debug = env.pvals["debug"] == 1
# 	for string in strings:
# 		print("storing '%s'" % string)
# 		b0 = env.cmap.char2bin(string[0])  # binary word associated with first character
# 		address = env.merge.merge(b0, b0)
# 		for char in string[1:]+"#":  # add stop character at end
# 			bchar = env.cmap.char2bin(char)
# 			if ma == "wx2":
# 				# store first part of character followed by xor part
# 				value = np.concatenate((bchar[0:wh_len], address[wh_len:]))
# 				assert len(value) == env.pvals["word_length"]
# 				if debug:
# 					print(" bchr:%s - %s" % (bina2str(bchar), char))
# 			else:
# 				value = bchar
# 			env.sdm.store(address, value)
# 			address = env.merge.merge(bchar, address)
# 		env.record_saved_string(string)

# def store_strings_old(env, strings):
# 	ma = env.pvals["merge_algorithm"]
# 	wl_half = int(env.pvals["word_length"]/2)
# 	start_bit = env.pvals["start_bit"]
# 	debug = env.pvals["debug"] == 1
# 	for string in strings:
# 		print("storing '%s'" % string)
# 		b0 = env.cmap.char2bin(string[0])  # binary word associated with first character
# 		address = env.merge.merge(b0, b0)
# 		for char in string[1:]+"#":  # add stop character at end
# 			bchar = env.cmap.char2bin(char)
# 			if ma == "wx2":
# 				if debug:
# 					print(" bchr:%s - %s" % (bina2str(bchar), char))
# 				# store .5 bchar + end of address (xor part)
# 				value = np.concatenate((bchar[0:wl_half], address[wl_half:]))
# 			else:
# 				value = bchar
# 			env.sdm.store(address, value)
# 			address = env.merge.merge(bchar, address)
# 		env.record_saved_string(string)


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

def reverse_recall(suffix, env):
	# recall string in reverse starting with suffix, works only with wx2 method
	found = [ prefix, ]
	word2 = prefix
	ma = env.pvals["merge_algorithm"]
	if ma != "wx2":
		print("Reverse recall only works with 'wx2' algorithm")
		return [found, word2]
	# build address to start searching
	wh_len = env.merge.pvals["wx"]["wh_len"]
	item_len = env.merge.pvals["wx"]["item_len"]
	word_length = env.pvals["word_length"]
	threshold = word_length * env.pvals["char_match_fraction"]
	max_num_recalled_chars = len(max(env.saved_strings, key=len)) + 5
	ma = env.pvals["merge_algorithm"]
	debug = env.pvals["debug"]
	b0 = env.cmap.char2bin(prefix[0])  # binary word associated with first character
	address = env.merge.merge(b0, b0)
	for char in prefix[1:]:
		b = env.cmap.char2bin(char)
		address = env.merge.merge(b, address)
	if ma  == "wx2":
		if len(suffix) > 1:
			# iteratively read value at address until converges (no decrease in hamming distance between xor part in
			# address and value read)
			pb_len = sum(env.merge.pvals["wx"]["bin_sizes"][0:len(prefix)]) # number prefix bits
			prev_address = address  # a large number so first found_value reduces hdiff
			hconverge=[]
			max_steps = 10
			while True:
				found_value = env.sdm.read(address)
				address = np.concatenate((address[0:pb_len], found_value[pb_len:]))
				hdiff = hamming(prev_address, address)
				hconverge.append(hdiff)
				if hdiff == 0 or len(hconverge) > max_steps:
					break
				prev_address = address
			f_non_zero = np.count_nonzero(found_value) / len(found_value)
			print("Iterative converge in %s steps, f_non_zero=%s, hdiff=%s" % (len(hconverge), f_non_zero, hconverge))
			if f_non_zero < 0.1:
				print("Failed to converge to valid data (found_value near zero)")
				return [found, word2]
			if debug:
				print(" found_value = %s" % bina2str(found_value))
				print(" created address used to start reading sequence:\n %s" % bina2str(address))
	computed_address = address
	# now read sequence in reverse order using address derived from prefix
	ccount = 0
	while True:
		b = env.sdm.read(address)
		recalled_xor_part = b[wh_len:]
		# get previous character from item in address
		top_matches = env.cmap.bin2char(address, match_bits=item_len)
		found.append(top_matches)
		found_char = top_matches[0][0]
		bchar = env.cmap.char2bin(found_char)
		# new_address = 

		if ma == "wx2":
			if debug:
				print(" reading, addr=%s, result b=%s" % (bina2str(address), bina2str(b)))
			# value read (b) has char part followed by xor part
			# bchar = np.concatenate((b[0:wl_half], b[0:wl_half]))
			recalled_xor_part = b[wh_len:]
			# computed_xor_part = computed_address[wh_len:]
			computed_xor_part = computed_address[wh_len:]
			xor_hamming = " xor hamming =%s" % hamming(computed_xor_part, recalled_xor_part)
			# following to use computed_xor_part for all subsequent recall, ignoring what is in sdm for xor part
			# computed_xor_part = xor_part
			match_bits = item_len
			# test replacing xor part of address with read xor part:
			# address = np.concatenate((address[0:wh_len], recalled_xor_part))
		else:
			# bchar = b
			xor_hamming = None
			match_bits = None
		top_matches = env.cmap.bin2char(b, match_bits=match_bits)
		found.append(top_matches)
		found_char = top_matches[0][0]
		if found_char == "#" or top_matches[0][1] > threshold:
			# found stop char, or hamming distance to top match is over threshold
			break
		# get full binary word corresponding to character
		if top_matches[0][1] > 0 or match_bits is not None:
			bchar = env.cmap.char2bin(found_char)
		else:
			bchar = b
		# at this point, have:
		#  address - address of value just read from memory
		#  b - value read using "address"
		#  found_char - found character (printable code) derived from b
		#  bchar - binary word corresponding to found_char
		# wx2 method only:
		#  recalled_xor_part - xor part read from memory (from b)
		#  computed_xor_part - xor part computed at each iteration using previous address and bchar
		# Now need to build the new address
		# create history to use in merge to create new address for reading next character in the sequence
		# if ma == "wx2":
		# 	# substitute computed 
		# 	history = np.concatenate((address[0:wh_len], computed_xor_part))
		# 	# if debug:
		# 	# 	print("computing address")
		# 	# computed_address = env.merge.merge(bchar, computed_address)
		# else:
		# 	history = address
		new_computed_address = env.merge.merge(bchar, computed_address)
		new_address = env.merge.merge(bchar, address)
		diff = np.count_nonzero(address!=new_address)
		found[-1].append("addr_diff=%s" % diff)
		if xor_hamming:
			found[-1].append(xor_hamming)
		if diff == 0:
			found[-1].append("found fixed point in address")
			    #, address=\n%s" % (bina2str(new_address)))
			break
		address = new_address
		computed_address = new_computed_address
		word2 += found_char
		ccount += 1
		if ccount > max_num_recalled_chars:
			break
	return [found, word2]


def reverse_recall_strings(env, strings):
	# recall in reverse each string in strings
	for suffix in strings:
		found, word2 = reverse_recall(suffix, env)
		pp.pprint(found)
		print ("Reverse recall '%s'" % suffix)
		print("found: '%s'" % word2)


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
		print("%s strings, %s errors" % (len(strings), error_count))

# def recall_strings_old(env, strings, prefix_mode = False):
# 	# recall list of strings starting with first character of each
# 	# if prefix_mode is True, use entire string as prefix for recall
# 	error_count = 0
# 	for word in strings:
# 		print ("\nRecall '%s'" % word)
# 		prefix = word if prefix_mode else word[0]
# 		found, word2 = recall(prefix, env)
# 		if word != word2:
# 			error_count += 1
# 			msg = "ERROR"
# 		else:
# 		    msg = "Match"
# 		pp.pprint(found)
# 		print ("Recall '%s'" % word)
# 		if not prefix_mode:
# 			print("found: '%s' - %s" % (word2, msg))
# 		else:
# 			print("found: '%s'" % word2)
# 	if not prefix_mode:
# 		print("%s strings, %s errors" % (len(strings), error_count))


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
