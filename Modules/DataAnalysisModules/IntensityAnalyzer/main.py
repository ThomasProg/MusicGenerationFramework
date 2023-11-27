from ctypes import cdll
import ctypes
import os
import matplotlib.pyplot as plt

# my_library = cdll.LoadLibrary('./my_library_wrapper.so')
my_library = cdll.LoadLibrary('C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Build/Modules/RuntimeModules/ParseMidiFileIntensity/Debug/ParseMidiFileIntensity.dll')

# Define a structure to match the C struct
class Intensities(ctypes.Structure):
    _fields_ = [("nbTracks", ctypes.c_int), ("data1", ctypes.c_void_p), ("data2", ctypes.c_void_p)]

class Vector(ctypes.Structure):
    _fields_ = [("length", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_int))]

# Define the argument type of the function
my_library.parseIntensities.argtypes = [ctypes.c_char_p]
# Set the return type of the function to be Vector
my_library.parseIntensities.restype = Intensities

my_library.getTrackVelocities.argtypes = [Intensities, ctypes.c_int]
my_library.getTrackVelocities.restype = Vector

my_library.getTrackTimings.argtypes = [Intensities, ctypes.c_int]
my_library.getTrackTimings.restype = Vector

# Define the path to the folder
folder_path = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Datasets/LakhMidi-full"

# intensities = None

# Loop over all files in the folder
# found = False
for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        # Check if the item is a file (not a subdirectory)
        if os.path.splitext(filename)[1] == ".mid":
            print(filename.encode("utf-8"))
            try:
                parsedIntensities = my_library.parseIntensities(os.path.join(dirpath, filename).encode("utf-8"))
                for trackIndex in range(0, parsedIntensities.nbTracks):
                    timings = my_library.getTrackTimings(parsedIntensities, trackIndex)
                    velocities = my_library.getTrackVelocities(parsedIntensities, trackIndex)

                    data = [velocities.data[i] for i in range(velocities.length)]
                    # data[:] = (value for value in data if value != 0)
                    # dates = [i for i in range(len(data))]
                    dates = [timings.data[i] for i in range(timings.length)]

                    if (velocities.length == 0):
                        continue

                    vel1 = data[0]
                    shouldSkip = True
                    for d in data:
                        if (d != vel1):
                            shouldSkip = False
                            break

                    if (shouldSkip):
                        continue

                    plt.figure(figsize=(10, 5))  # Set the figure size (optional)
                    plt.plot(dates, data, marker='o')  # 'o' adds points at data

                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.title(filename + " / Track : " + str(trackIndex))

                plt.show()
            
            except:
                print("Couldn't parse file")

                # if (intensities.length == 3):
                #     found = True
                #     break
    # if found:
    #     break

# intensities = my_library.parseIntensities(b"C:/Users/thoma/Downloads/archive/Ludwig_van_Beethoven/Fur_Elise.1.mid")
# intensities = my_library.parseIntensities(b"C:/Users/thoma/Downloads/archive/Rick_Astley/Never_Gonna_Give_You_Up.mid")
# intensities = my_library.parseIntensities(b"C:/Users/thoma/Downloads/archive/.38 Special/Caught Up In You.mid")

# dates = [i for i in range(intensities.length)]
# data = [intensities.data[i] for i in range(intensities.length)]

# plt.figure(figsize=(10, 5))  # Set the figure size (optional)
# plt.plot(dates, data, marker='o')  # 'o' adds points at data

# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.title('Time Series Data')

# plt.show()  

# my_library.deleteIntensities(intensities)


