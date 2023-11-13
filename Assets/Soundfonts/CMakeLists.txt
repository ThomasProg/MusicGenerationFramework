cmake_minimum_required(VERSION 3.10)

include(FetchContent)

function(download_soundfont soundfontName url)

project("${soundfontName}")

message(STATUS "Downloading ${soundfontName} Soundfont...")
Set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare("${soundfontName}"
    URL ${url}
    DOWNLOAD_NO_EXTRACT true
    TLS_VERIFY true
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${soundfontName}"
    # DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_Populate("${soundfontName}")

endfunction(download_soundfont)

# WTFPL（Do What The Fuck You Want To Public License）

# https://musical-artifacts.com/artifacts/538
download_soundfont(
    Roland_SC88
    https://musical-artifacts.com/artifacts/538/Roland_SC-88.sf2
)

# https://musical-artifacts.com/artifacts/1790
download_soundfont(
    CASIO_CTK-533_Piano1_00_
    https://musical-artifacts.com/artifacts/1790/CTK-533_Piano1__00___Casio_.sf2
)

# https://musical-artifacts.com/artifacts/1558
download_soundfont(
    FridayNightFunkin
    https://musical-artifacts.com/artifacts/1558/FNF.sf2
)


# Public Domain License

# https://musical-artifacts.com/artifacts/914
download_soundfont(
    Undertale
    https://musical-artifacts.com/artifacts/914/undertale.sf2
)

# https://musical-artifacts.com/artifacts/371
download_soundfont(
    RhodesEVP73
    https://musical-artifacts.com/artifacts/371/rhodes_73.sf2
)

# https://musical-artifacts.com/artifacts/1977
download_soundfont(
    OldWurlitzerUpright
    https://musical-artifacts.com/artifacts/1977/Discarded_Wurlitzer_piano.sf2
)


# Creative Commons BY 4.0 International

# https://musical-artifacts.com/artifacts/433
download_soundfont(
    Touhou
    https://musical-artifacts.com/artifacts/433/Touhou.sf2
)

# https://musical-artifacts.com/artifacts/433
download_soundfont(
    ColomboGMGS2
    https://musical-artifacts.com/artifacts/1234/ColomboGMGS2__SF2_.zip
)


