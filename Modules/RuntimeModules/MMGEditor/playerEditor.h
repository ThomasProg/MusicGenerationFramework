#pragma once

#include "Player.h"
#include <thread>

class PlayerEditor
{
	Player player;
	std::string sfPath;
	std::string midiPath;

	std::thread music;

public:
	PlayerEditor();
	void Render();
};