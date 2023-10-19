#ifndef _MIDI_HELPERS_H_
#define _MIDI_HELPERS_H_

#include <cstdint>

#ifdef MAKE_DLL // CPP
#define MIDIHELPERS __declspec(dllexport)
#else
#define MIDIHELPERS __declspec(dllimport)
#endif

enum class MIDIHELPERS ENoteEvent : uint8_t
{
    // Regular Events
    NOTE_OFF         = 0x80,
    NOTE_ON          = 0x90,
    NOTE_AFTER_TOUCH = 0xA0, 
    CONTROL_CHANGE   = 0xB0, 
    PGM_CHANGE       = 0xC0,
    CHANNEL_AFTER_TOUCH = 0xD0,
    PITCH_BEND = 0xE0,
};

enum class MIDIHELPERS EMidiMeta : uint8_t
{
    SEQ_NUM         = 0x00,
    TEXT            = 0x01,
    COPYRIGHT       = 0x02,
    TRACK_NAME      = 0x03,
    INSTRUMENT_NAME = 0x04,
    LYRICS          = 0x05,
    MARKER           = 0x06,
    CUE_POINT       = 0x07,
    CHANNEL_PREFIX  = 0x20,
    MIDI_PORT       = 0x21,
    END_OF_TRACK    = 0x2F,
    SET_TEMPO       = 0x51,
    SMPTE_OFFSET    = 0x54,
    TIME_SIGNATURE  = 0x58,
    KEY_SIGNATURE   = 0x59,
    SEQ_SPECIFIC    = 0x7F,
};

enum class MIDIHELPERS EControlChange : uint8_t
{
    BANK_SELECT = 0,
    MODULATION_WHEEL = 1,
    BREATH_CONTROLLER = 2,

    FOOT_PEDAL = 4,
    PORTAMENTO_TIME = 5,
    DATA_ENTRY = 6,
    VOLUME = 7,
    BALANCE = 8,

    PAN = 10,
    EXPRESSION = 11,      
    EFFECT_CONTROLLER_1 = 12,
    EFFECT_CONTROLLER_2 = 13,
    GENERAL_PURPOSE_1 = 16,
    GENERAL_PURPOSE_2 = 17,
    GENERAL_PURPOSE_3 = 18,
    GENERAL_PURPOSE_4 = 19,  

    DAMPER_PEDAL = 64,
    PORTAMENTO_ON_OFF = 65,
    SOSTENUTO_PEDAL_ON_OFF = 66,  
    SOFT_PEDAL_ON_OFF = 67,
    LEGATO_FOOT_SWITCH = 68,
    HOLD_2 = 69,
    SOUND_CONTROLLER_1 = 70,
    SOUND_CONTROLLER_2 = 71,
    SOUND_CONTROLLER_3 = 72,
    SOUND_CONTROLLER_4 = 73,
    SOUND_CONTROLLER_5 = 74,
    SOUND_CONTROLLER_6 = 75,
    SOUND_CONTROLLER_7 = 76,
    SOUND_CONTROLLER_8 = 77,
    SOUND_CONTROLLER_9 = 78,
    SOUND_CONTROLLER_10 = 79,
    GENERAL_PURPOSE_MIDI_CC_CONTROLLER_1 = 80,
    GENERAL_PURPOSE_MIDI_CC_CONTROLLER_2 = 81,
    GENERAL_PURPOSE_MIDI_CC_CONTROLLER_3 = 82,
    GENERAL_PURPOSE_MIDI_CC_CONTROLLER_4 = 83,
    PORTAMENTO_CC_CONTROL = 84,

    HIGH_RESOLUTION_VELOCITY_FIX = 88,

    EFFECT_1_DEPTH = 91,
    EFFECT_2_DEPTH = 92,
    EFFECT_3_DEPTH = 93,
    EFFECT_4_DEPTH = 94,
    EFFECT_5_DEPTH = 95,

    DATA_INCREMENT = 96,
    DATA_DECREMENT = 97,
    NON_REGISTERED_PARAMETER_NUMBER_LSB = 98,   
    NON_REGISTERED_PARAMETER_NUMBER_MSB = 99,
    REGISTERED_PARAMETER_NUMBER_LSB = 100,   
    REGISTERED_PARAMETER_NUMBER_MSB = 101,

    ALL_SOUND_OFF = 120,
    RESET_ALL_CONTROLLERS = 121,
    LOCAL_ON_OFF_SWITCH = 122,
    ALL_NOTES_OFF = 123,
    OMNI_MODE_OFF = 124,
    OMNI_MODE_ON = 125,
    MONO_MODE = 126,
    POLY_MODE = 127,
};

// https://users.cs.cf.ac.uk/dave/Multimedia/node158.html
enum class MIDIHELPERS ESysEvent : uint8_t
{
    // System exclusive messages (i.e. non standard)
    SYSTEM_EXCLUSIVE = 0xF0,

    // System common messages
    MIDI_TIMING_CODE = 0xF1,    // has 1 param
    SONG_POSITION_POINTER = 0xF2, // has 2 params
    SONG_SELECT = 0xF3, // has no params
    TUNE_REQUEST = 0xF6, 

    ESCAPE_EVENT = 0xf7,

    // System real-time messages
    TIMING_CLOCK = 0xF8,
    START_SEQUENCE = 0xFA,
    CONTINUE_SEQUENCE = 0xFB,
    STOP_SEQUENCE = 0xFC,
    ACTIVE_SENSING = 0xFE,
    SYSTEM_RESET = 0xFF,
};

#endif