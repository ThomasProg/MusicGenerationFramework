

from transformers import GPT2TokenizerFast

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

# # vocab=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
# #        "\nTimeShift: ",
# #        "\nPitch: ",
# #        "\nDuration: ",
# #        "it"
# # ]

# vocab={"0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, 
#        "\nTimeShift: ":10,
#        "\nPitch: ":11,
#        "\nDuration: ":12,
# }

# tokenizer = Tokenizer(models.BPE(vocab, merges = vocab))


inputStr = """
TimeShift: 720
Pitch: 67
Duration: 683"""

# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
# print("Pre tokenize:", tokenizer.pre_tokenizer.pre_tokenize_str(inputStr))



# trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"], initial_alphabet=vocab)
# tokenizer.train_from_iterator(inputStr, trainer=trainer)


# # vocab = tokenizer.get_vocab()
# # print("vocab: ", vocab)

# # tokenizer.model = models.BPE()


# encoding = tokenizer.encode(inputStr)
# print("Encode:", encoding.tokens)



# tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
# encoding = tokenizer.encode(inputStr)
# # start, end = encoding.offsets[4]
# # print(inputStr[start:end])



# tokenizer.decoder = decoders.ByteLevel()
# print("Decode:", tokenizer.decode(encoding.ids))



# from transformers import PreTrainedTokenizerFast

# wrapped_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer,
#     bos_token="<|endoftext|>",
#     eos_token="<|endoftext|>",
# )

event_tokens=[   
    "\nTimeShift: ",
    "\nPitch: ",
    "\nDuration: ",
    ]

numbers_tokens=[]
# for each midi note
for i in range(0, 127): 
    numbers_tokens.append(str(i))

special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",]

dict = {}

def addListToDict(list):
    for l in list:
        dict[l] = len(dict)

addListToDict(special_tokens)
addListToDict(numbers_tokens)
addListToDict(event_tokens)


# base_vocabDict={"0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, 
#        "\nTimeShift: ":10,
#        "\nPitch: ":11,
#        "\nDuration: ":12,
# }

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a tokenizer with the base vocabulary
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]", vocab=dict))
# tokenizer = Tokenizer(models.BPE())

def displayVocab(tok):
    vocab = tok.get_vocab()
    vocab = sorted(vocab.items(), key=lambda item: item[1])
    print("vocab: ", vocab)

displayVocab(tokenizer)


# trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=base_vocab + ["<|endoftext|>"])

# tokenizer.train_from_iterator([inputStr], trainer=trainer)
# tokenizer.train_from_iterator([inputStr])

encoding = tokenizer.encode(inputStr)
print("Encode:", encoding.tokens)

# displayVocab(tokenizer)