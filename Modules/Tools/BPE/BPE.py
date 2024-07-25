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
    # 2 -> "a"
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

    def add_token_for_sequence(self, newSequence: tuple):
        if not(newSequence in self.compressVocab):
            self.compressVocab[newSequence] = self.vocab_size()
            self.isDirty = True

    def add_tokens_for_sequences(self, newSequences):
        for newSequence in newSequences:
            self.add_token_for_sequence(newSequence)

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
    
    def get_most_frequent_pair(self, tokensList):
        pairToCount = {}
        for i in range(len(tokensList) - 1):
            newSequence:tuple = tokensList[i] + tokensList[i+1] 
            if newSequence in pairToCount:
                pairToCount[newSequence] += 1
            else:
                pairToCount[newSequence] = 1

        sortedMap = dict(sorted(pairToCount.items(), key=lambda item: item[1], reverse=True))
        it = iter(sortedMap.items())
        print(sortedMap)
        return next(it)

    # def sequence_to_token(self, sequence):
    #     sorted(self.compressVocab, key=len)
    #     for i in range(len(sequence)):
    #         for vocab in self.compressVocab:
    #             j = 0
    #             while (j < len(vocab) and (i+j) < len(sequence) and sequence[i+j] == vocab[j]):
    #                 j += 1

    #             # if fits
    #             if (j == len(vocab)):
    #                 break

    #             print(len(inTokens[0]))
        
    #     return inTokens

    # def sequence_to_vocabs(self, sequence):
    #     sorted(self.compressVocab, key=len, reverse=True)
    #     for vocab in self.compressVocab:
    #         print("vocab: ", vocab)
    #         print("sequence length: ", len(sequence))
    #         print(sequence)
    #         for i in range(len(sequence)):
    #             # print(i)
    #             j = 0
    #             while (j < len(vocab) and (i+j) < len(sequence) and sequence[i+j] == vocab[j]):
    #                 j += 1

    #             # if fits
    #             if (j != len(vocab)):
    #                 break

    #             # print(sequence[:i])
    #             # print(sequence[i+j:])
    #             newSequence = sequence[:i-1]
    #             newSequence.append(vocab)
    #             newSequence.extend(sequence[i+j+1:])
    #             sequence = newSequence


    #             # print("fitting! ", vocab)
    #             # sequence[i:i+j] = [vocab]
        
    #     return sequence

    @staticmethod
    def replaceRange(list1, element, start, end):
        newSequence = []

        for k in range(0, start):
            newSequence.append(list1[k])

        newSequence.append(element)

        for k in range(end, len(list1)):
            newSequence.append(list1[k])

        return newSequence
    
    @staticmethod
    def replaceSubListToElem(list1, fromSubList, element):
        for i in range(len(list1)):
            j = 0
            # print()
            # print(list1[0][0])
            # print(fromSubList[0])
            while (j < len(fromSubList) and (i+j) < len(list1) and list1[i+j] == fromSubList[j]):
                j += 1

            # if fits
            if (j != len(fromSubList)):
                continue

            # print("replacing")
            list1 = BPE.replaceRange(list1, element, i, i+j+1)

        return list1

    def sequence_to_vocabs(self, sequence):
        sorted(self.compressVocab, key=len, reverse=True)

        for vocab in self.compressVocab:
            sequence = BPE.replaceSubListToElem(sequence, vocab, vocab)
        
        return sequence


    def train_from_iterator(self, sequence, trainer: BPETrainer):
        print(sequence)
        inTokens = self.sequence_to_vocabs(sequence)
        print(inTokens)
        
        # key, value = self.get_most_frequent_pair(inTokens)
        # print("Fusing : ", key)
        # print("frequency: ", value / len(inTokens))

        # self.add_token_for_sequence(key)
