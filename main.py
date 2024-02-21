
pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

# vocab objects are objects that "maps each word in the problem context to a 0-based integer index, based on how
# common the word is" (Mccaffrey, 2021) from build_vocab, load_vocab is a static method that loads a vocab pickle
# build_vocab should take care of the basic vocab object set up for us
