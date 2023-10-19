#include <vector>

typedef struct {
    int length;
    int* data;
} __declspec(dllexport) VectorPtr;

typedef struct {
    int nbTracks; 

    std::vector<std::vector<int>>* velocitiesPerTrack;
    std::vector<std::vector<int>>* timesPerTrack;

} __declspec(dllexport) Intensities;


extern "C" {
    // Define a function that returns a pointer to a vector of integers.
    __declspec(dllexport) Intensities parseIntensities(const char* filepath);

    // Define a function that takes a pointer to a vector and deletes it.
    __declspec(dllexport) void deleteIntensities(Intensities vec);

    __declspec(dllexport) VectorPtr getTrackVelocities(Intensities vec, int track);
    __declspec(dllexport) VectorPtr getTrackTimings(Intensities vec, int track);
}