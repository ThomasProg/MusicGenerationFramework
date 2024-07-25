from BPE import *


tok = BPE()

# All ascii characters
# for i in range(0, 127):
#     # ord('a')
#     tok.add_tokens(i)

# for i in range(0, 127):
#     tok.add_token_for_sequence((i,))

for i in range(127, -1, -1):
    tok.add_token_for_sequence((i,))


print(tok.compressVocab)
print("size : ", tok.vocab_size())

def tokStr(string: str) -> list:
    out = []
    for character in string:
        out.append(ord(character))
    return out

def untokStr(tokens) -> str:
    out = ""
    for token in tokens:
        out += chr(token)
    return out

print(untokStr(tokStr("Hello world!")))

corpus = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua", 
          "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat"]

tuples_list = [(ord(char),) for char in corpus[0]]

print("========= TRAINING =========")

# print(BPE.replaceRange("hello", "yo", 1, 3))


trainer = BPETrainer()
tok.rebuildUncompressVocab()
tok.train_from_iterator(tuples_list, trainer)

