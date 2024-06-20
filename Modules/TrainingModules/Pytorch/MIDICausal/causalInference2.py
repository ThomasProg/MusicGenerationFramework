# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = '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'










# Inference

prompt = "Let me bump! This is nonsense!"


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Progz/MIDICausalFinetuning2")
inputs = tokenizer(prompt, return_tensors="pt").input_ids

import torch
# inputs2 = torch.tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
# inputs2 = torch.tensor([[60, 62, 58, 60]])
inputs2 = torch.tensor([[0]])


# from transformers import pipeline

# generator = pipeline("text-generation", model="Progz/MIDICausalFinetuning2")
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# generator(inputs)






from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Progz/MIDICausalFinetuning2")
outputs = model.generate(inputs2, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)


# print(outputs)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


quit()



import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi.containers import Note, Instrument, TempoChange

# Create a new MIDI file with a single track
midi_obj = miditoolkit.midi.parser.MidiFile()

# Define tempo (BPM)
tempo = 120  # 120 beats per minute
midi_obj.tempo_changes.append(TempoChange(tempo, 0))

# Define an instrument (0 is Acoustic Grand Piano)
instrument = Instrument(program=0, is_drum=False, name="Piano")


notes = []
for i, n in  enumerate(outputs[0]):
    notes.append((max(40, min(100, n.item())), i * 240, 240, 100))
# # Define a simple melody (C4, D4, E4, F4, G4) with durations and velocities
# notes = [
#     (60, 0, 480, 100),  # C4, start at tick 0, duration 480 ticks, velocity 100
#     (62, 480, 480, 100), # D4, start at tick 480, duration 480 ticks, velocity 100
#     (64, 960, 480, 100), # E4, start at tick 960, duration 480 ticks, velocity 100
#     (65, 1440, 480, 100),# F4, start at tick 1440, duration 480 ticks, velocity 100
#     (67, 1920, 480, 100) # G4, start at tick 1920, duration 480 ticks, velocity 100
# ]







# Add notes to the instrument
for note_number, start_tick, duration, velocity in notes:
    note = Note(
        start=start_tick,
        end=start_tick + duration,
        pitch=note_number,
        velocity=velocity
    )
    instrument.notes.append(note)

# Add the instrument to the MIDI file
midi_obj.instruments.append(instrument)

# Save the MIDI file
output_path = 'simple_melody.mid'
midi_obj.dump(output_path)