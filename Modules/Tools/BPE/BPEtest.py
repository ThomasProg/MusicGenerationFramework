from BPE import *


tok = BPE()

# All ascii characters
# for i in range(0, 127):
#     # ord('a')
#     tok.add_tokens(i)

tok.add_tokens(range(0, 127))

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

sentence = list(corpus[0])

trainer = BPETrainer()
tok.train_from_iterator(sentence, trainer)

