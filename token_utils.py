import os
import random
import json
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def token_setup(tokenizer):
    token_strings = [tokenizer.decode([i]) for i in range(50257)]
    num_tokens = len(token_strings)
    print(f"There are {num_tokens} tokens.")

    all_rom_tokens = []			#initialise list of all-roman tokens
    all_rom_token_indices = []

    for i in range(num_tokens):
      all_rom = True                       # Is token_string[i] all roman characters? Assume to begin that it is.
      for ch in range(len(token_strings[i])):
        if token_strings[i][ch] not in ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
          all_rom = False
      if all_rom == True and token_strings[i] not in [' ', '  ', '   ', '    ', '     ', '      ', '       ', '        ']:  # eliminate all the all-space tokens
        all_rom_tokens.append(token_strings[i])
        all_rom_token_indices.append(i)

    print(f"There are {len(all_rom_tokens)} all-Roman tokens.")

    all_rom_token_gt2_indices = [idx for idx in all_rom_token_indices if len(token_strings[idx].strip()) > 2]
    print(f"There are {len(all_rom_token_gt2_indices)} all-Roman tokens with more than two letters.")

    return token_strings, all_rom_token_indices, all_rom_token_gt2_indices


# THIS MAKES A LIST OF THE 50257 SIGNIFICANT TOKENS, AS WELL AS  SUB-LISTS OF ALL-ROMAN TOKENS
def load_token_strings_etc(tokenizer):

		num_tokens = len(token_strings)

		# The leading spaces in these token strings are weird, they somehow delete themselves and the character before whatever they get appended to
		#...so you got stuff like "The string'petertodd' spelled in all capital letters..."
		#...rather than "The string ' petertodd' spelled is..." as it should be
		# This seems to be an easy fix. Just go through the lists and replace the first character with an actual space!
		for token in token_strings:
			token = " " + token[1:]

		print(f"There are {num_tokens} tokens.")

		all_rom_tokens = []			#initialise list of all-roman tokens
		all_rom_token_indices = []

		for i in range(num_tokens):
			all_rom = True                       # Is token_string[i] all roman characters? Assume to begin that it is.
			for ch in range(len(token_strings[i])):
				if token_strings[i][ch] not in ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
					all_rom = False
			if all_rom == True and token_strings[i] not in [' ', '  ', '   ', '    ', '     ', '      ', '       ', '        ']:
				all_rom_tokens.append(token_strings[i])
				all_rom_token_indices.append(i)

		all_rom_token_gt2_indices = [idx for idx in all_rom_token_indices if len(token_strings[idx].lstrip()) > 2]


		print(f"There are {len(all_rom_tokens)} all-Roman tokens.")
		print(f"There are {len(all_rom_token_gt2_indices)} all-Roman tokens with more than two letters.")
	
		return token_strings, all_rom_tokens, all_rom_token_indices, all_rom_token_gt2_indices
