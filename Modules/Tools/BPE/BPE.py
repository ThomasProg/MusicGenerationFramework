from typing import Any


class BPETrainer:
    vocab_size = None # (int, optional) # The size of the final vocabulary, including all tokens and alphabet.
    min_frequency = None # (int, optional) # The minimum frequency a pair should have in order to be merged.
    show_progress = None # (bool, optional) # Whether to show progress bars while training.
    special_tokens = None # (List[Union[str, AddedToken]], optional) # A list of special tokens the model should know of.
    limit_alphabet = None # (int, optional) # The maximum different characters to keep in the alphabet.
    initial_alphabet = None # (List[str], optional) # A list of characters to include in the initial alphabet, even if not seen in the training dataset. If the strings contain more than one character, only the first one is kept.
    continuing_subword_prefix = None # (str, optional) # A prefix to be used for every subword that is not a beginning-of-word.
    end_of_word_suffix = None # (str, optional) # A suffix to be used for every subword that is a end-of-word.
    max_token_length = None # (int, optional) # Prevents creating tokens longer than the specified size. This can help with reducing polluting your vocabulary with highly repetitive tokens like ====== for wikipedia






class BPE:
    # "a" -> 2
    compressVocab = {}
    uncompressVocab = {}
    isDirty = False

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def rebuildUncompressVocab(self):
        self.uncompressVocab.clear()
        for key in self.compressVocab:
            self.uncompressVocab[self.compressVocab[key]] = key

        self.isDirty = False

    def vocab_size(self):
        return len(self.compressVocab)

    def add_token(self, newToken):
        if not(newToken in self.compressVocab):
            self.compressVocab[newToken] = self.vocab_size()
            self.isDirty = True

    def add_tokens(self, newTokens):
        for newToken in newTokens:
            self.add_token(newToken)

    # def add_special_tokens(self, newTokens, specialTokensDict):
    #     pass

    # def batch_decode(self, tokenIDsList) -> list:
    #     pass

    def decode(self, inTokenIDs) -> list:
        if (self.isDirty):
            self.rebuildUncompressVocab()

        outTokens = []
        for token in inTokenIDs:
            outTokens.append(self.uncompressVocab[token])

        return outTokens

    def encode(self, inTokens) -> list:
        outTokenIDs = []
        for token in inTokens:
            outTokenIDs.append(self.compressVocab[token])

        return outTokenIDs
    
    def train_from_iterator(self, tokensList, trainer: BPETrainer):
        pairToCount = {}
        for i in range(len(tokensList) - 1):
            pair = (tokensList[i], tokensList[i+1])
            if pair in pairToCount:
                pairToCount[pair] += 1
            else:
                pairToCount[pair] = 1

        sortedMap = dict(sorted(pairToCount.items(), key=lambda item: item[1], reverse=True))
        print(sortedMap)
        print()
        it = iter(sortedMap.items())
        key, value = next(it)
        print("frequency: ", value / len(tokensList))

        self.add_token(key)