# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = ''

# Inference

from sequentialTokenizer import MIDISequentialTokenizer
tokenizer = MIDISequentialTokenizer()

import torch

# inputs2 = torch.tensor([[51]])
# inputs2 = torch.tensor([[83]])
# inputs2 = torch.tensor([[80]])
# inputs2 = torch.tensor([[65]])


inputs2 = None

if (tokenizer.addPitchTokens):
    inputs2 = torch.tensor([[tokenizer.pitchToToken(60)]])    
# inputs2 = torch.tensor([[tokenizer.bos_token_id]])

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Progz/SequentialFinetuningFromFolder")
# model = AutoModelForCausalLM.from_pretrained("Progz/MIDICausalFinetuning3_5thsymphony")
# outputs = model.generate(inputs2, max_new_tokens=6*30-1, do_sample=True, top_k=50, top_p=0.95)
# outputs = model.generate(inputs2, max_new_tokens=1024//2//2-1, do_sample=True, top_k=50, top_p=0.95, pad_token_id=2000)

from transformers import LogitsProcessorList, LogitsWarper, LogitsProcessor

# Cant use with generate
class CustomLogitsWarper(LogitsWarper):
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # Apply the custom function to the logits
        # custom_scores = self.custom_function(scores)
        print("CustomLogitsWarper::call")
        return scores
    
class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # Apply the custom function to the logits
        # custom_scores = self.custom_function(scores)
        # print("CustomLogitsProcessor::call")
        # print(scores)

        global tokenizer

        scores = torch.softmax(scores, dim=-1)
        # print("normalized scores:")
        # print(scores)

        # total = 0
        # for v in scores[0]:
        #     total += v

        # for i in range(tokenizer.firstPitchDeltaToken(), tokenizer.lastPitchDeltaToken() + 1 - 20):
        #     scores[0][i] = 0

        # print("total: ", total)

        scores = torch.log(scores)

        return scores

outputs = model.generate(inputs2, logits_processor=LogitsProcessorList([CustomLogitsProcessor()]), max_new_tokens=1024//2-1, do_sample=True, top_k=50, top_p=0.95, pad_token_id=2000)


torch.set_printoptions(threshold=10_000)
print(outputs[0])
midiFile = tokenizer.decode(outputs[0].tolist())
# midiFile.ticks_per_beat = 48
midiFile.ticks_per_beat = 4*4
for note in midiFile.instruments[0].notes:
    # note.pitch = min(max(127, note.pitch), 0)
    print("%d / %d / %d" % (note.start, note.end, note.pitch))
midiFile.dump("gentest.mid")






