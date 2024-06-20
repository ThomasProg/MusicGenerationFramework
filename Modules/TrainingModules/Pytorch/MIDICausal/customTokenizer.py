

from transformers import GPT2TokenizerFast

# VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# Instantiate the tokenizer from the configuration
tokenizer = GPT2TokenizerFast(
    vocab_file="./customTok/vocab.json",
    merges_file="./customTok/merges.txt",
    tokenizer_file="./customTok/tokenizer.json")

tokenizer.save_pretrained("./customTok/")

# vocab_file (`str`, *optional*):
#     Path to the vocabulary file.
# merges_file (`str`, *optional*):
#     Path to the merges file.
# tokenizer_file (`str`, *optional*):
#     Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
#     contains everything needed to load the tokenizer.
# unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
#     The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
#     token instead.
# bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
#     The beginning of sequence token.
# eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
#     The end of sequence token.
# add_prefix_space (`bool`, *optional*, defaults to `False`):
#     Whether or not to add an initial space to the input. This allows to treat the leading word just as any
#     other word. (GPT2 tokenizer detect beginning of words by the preceding space).