#pragma once

#include "Player.h"
#include <future>

class PlayerEditor
{
	Player player;
	std::string sfPath;
	std::string midiPath;
	std::future<void> music;

public:
	PlayerEditor();
	void Render();
};