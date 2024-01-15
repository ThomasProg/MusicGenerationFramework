#include "FluidsynthPlayerAsync.h"
#include "MIDIPlayerAsync.h"
#include "FluidsynthMIDIPlayer.h"

FluidsynthPlayerAsync::FluidsynthPlayerAsync(class MIDIMusic* music)
{
    fluidsynthPlayer.music = music;
    player = &fluidsynthPlayer;
}

class FluidsynthPlayerAsync* FluidsynthPlayerAsync_Create(class MIDIMusic* music, const char* soundfontPath)
{
    FluidsynthPlayerAsync* player = new FluidsynthPlayerAsync(music); 
    player->fluidsynthPlayer.LoadSoundfont(soundfontPath);
    return player;
}

void FluidsynthPlayerAsync_Destroy(class FluidsynthPlayerAsync* player)
{
    if (player)
    {
        player->Stop();
        delete player;
    }
}

void FluidsynthPlayerAsync_Play(FluidsynthPlayerAsync* player)
{
    player->Play();
}

void FluidsynthPlayerAsync_PlayAsync(FluidsynthPlayerAsync* player)
{
    player->playerFuture = std::async(std::launch::async, FluidsynthPlayerAsync_Play, player);
}