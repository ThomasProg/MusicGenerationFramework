from ctypes import cdll
import ctypes
import os
import matplotlib.pyplot as plt

# my_library = cdll.LoadLibrary('./my_library_wrapper.so')
my_library = cdll.LoadLibrary('C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Build/Modules/RuntimeModules/PlayMIDI/Debug/PlayMIDI.dll')
# my_library = cdll.LoadLibrary('C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Build/Modules/RuntimeModules/EasyMidiFileParserCpp/Debug/EasyMidiFileParserCpp.dll')


# # Define a structure to match the C struct
# class MIDINote(ctypes.Structure):
#     _fields_ = [("start", ctypes.c_int), 
#                 ("end", ctypes.c_int), 
#                 ("channel", ctypes.c_int), 
#                 ("key", ctypes.c_int), 
#                 ("velocity", ctypes.c_int)]

class NoteOn(ctypes.Structure):
    _fields_ = [("start", ctypes.c_int), 
                ("channel", ctypes.c_int), 
                ("key", ctypes.c_int), 
                ("velocity", ctypes.c_int),
                ("isValid", ctypes.c_bool)]
    
class ProgramChange(ctypes.Structure):
    _fields_ = [("start", ctypes.c_int), 
                ("channel", ctypes.c_int), 
                ("newProgram", ctypes.c_int), 
                ("isValid", ctypes.c_bool)]

class Vector(ctypes.Structure):
    _fields_ = [("length", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_void_p))]

my_library.CreateConvertingParser.argtypes = []
my_library.CreateConvertingParser.restype = ctypes.c_void_p

# Define the argument type of the function
my_library.ParseFromParser.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

my_library.DeleteMIDIParser.argtypes = [ctypes.c_void_p]
# my_library.DeleteMIDIParser.restype = Vector

my_library.GetNbTracksFromParser.argtypes = [ctypes.c_void_p]
my_library.GetNbTracksFromParser.restype = ctypes.c_int32

my_library.GetNotesFromTrack.argtypes = [ctypes.c_void_p, ctypes.c_int32]
my_library.GetNotesFromTrack.restype = Vector

my_library.CastToNoteOn.argtypes = [ctypes.c_void_p]
my_library.CastToNoteOn.restype = NoteOn

my_library.CastToProgramChange.argtypes = [ctypes.c_void_p]
my_library.CastToProgramChange.restype = ProgramChange

# Define the path to the folder
# folder_path = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-full"
folder_path = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean"

# Loop over all files in the folder
# found = False
r = 0
for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        if (filename != "I_Do.mid"):
            continue

        # Check if the item is a file (not a subdirectory)
        if os.path.splitext(filename)[1] == ".mid":
            filepath = (dirpath + "/" + filename).encode("utf-8")
            # print(filename.encode("utf-8"))
            print(filepath)

            parser = my_library.CreateConvertingParser()
            # my_library.ParseFromParser(parser, b"C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/Blondie/Call_Me.2.mid")
            # my_library.ParseFromParser(parser, b"C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/.38 Special/Caught Up In You.mid")
            # my_library.ParseFromParser(parser, b"C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-Clean/.38 Special/Fantasy Girl.mid")
            
            try:
                my_library.ParseFromParser(parser, filepath)
            except:
                print("Parsing failed.")

            nbTracks = my_library.GetNbTracksFromParser(parser)

            for trackIndex in range(nbTracks):
                notesVec = my_library.GetNotesFromTrack(parser, trackIndex)
                times = []
                notes = []
                colors = []
                program = 0
                percussive = False
                for j in range(notesVec.length):
                    note = my_library.CastToNoteOn(notesVec.data[j])
                    if (note.isValid):
                        times.append(note.start)
                        notes.append(note.key)
                        if (program >= 113 and program <= 120):
                            percussive = True
                            colors.append('red')
                        else:
                            colors.append('blue')
                    else:
                        programChange = my_library.CastToProgramChange(notesVec.data[j])
                        if (programChange.isValid):
                            program = programChange.newProgram
                            print("New Program : " + str(program))
                            print(programChange.start)

                percussive = True
                if (percussive):
                        plt.figure(figsize=(10, 5))  # Set the figure size (optional)
                        # plt.plot(times, notes, marker='o')  # 'o' adds points at data
                        plt.scatter(times, notes, marker='o', color = colors)

                        plt.xlabel('Time')
                        plt.ylabel('Note')
                        plt.title(filename + " / Track : " + str(trackIndex))

                        plt.show()

            my_library.DeleteMIDIParser(parser)



