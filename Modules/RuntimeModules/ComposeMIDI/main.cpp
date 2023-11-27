#include <SFML/Window/Event.hpp>
#include <SFML/Window/Window.hpp>
#include <future>
#include <thread>
#include <iostream>
#include "MIDIEvents.h"
#include "Player.h"

int main()
{
    sf::Window window(sf::VideoMode(800, 600), "My window");
    sf::Event event;

    Player player;

    std::string sfPath = "C:/Users/thoma/PandorasBox/Projects/ModularMusicGenerationModules/Assets/Soundfonts/Touhou/Touhou.sf2";
    int sf = player.LoadSoundfont(sfPath.c_str());

    player.notesPerTrack.resize(1);
    // player.Play();

     auto future = std::async(std::launch::async, [&player]
     { 
         player.Play();
     });

     std::vector<sf::Keyboard::Scancode> inputs =
     {
        sf::Keyboard::Scan::S,
        sf::Keyboard::Scan::D,
        sf::Keyboard::Scan::F,
        sf::Keyboard::Scan::J,
        sf::Keyboard::Scan::K,
        sf::Keyboard::Scan::L,
        sf::Keyboard::Scan::Space
     };

     std::vector<int> notes =
     {
         57,
         59,
         60,
         62,
         64,
         65,
         67
     };

    while (1)
    {
        // while there are pending events...
        while (window.pollEvent(event))
        {
            int i = 0;

            // check the type of the event...
            switch (event.type)
            {
                // window closed
                case sf::Event::Closed:
                    window.close();
                    break;

                // key pressed
                case sf::Event::KeyPressed:
                    // if (event.key.scancode == sf::Keyboard::Scan::Escape)
                    for (sf::Keyboard::Scancode& scan : inputs)
                    {
                        if (event.key.scancode == scan)
                        {
                            NoteOn* note = new NoteOn();

                            note->start = player.GetPlayerTime();
                            note->channel = 0;
                            note->key = notes[i];// 60;
                            note->velocity = 90;

                            player.notesPerTrack.resize(1);
                            player.InsertEvent(0, note);
                            std::cout << "the escape key was pressed" << std::endl;
                            std::cout << "scancode: " << event.key.scancode << std::endl;
                            std::cout << "code: " << event.key.code << std::endl;
                            std::cout << "control: " << event.key.control << std::endl;
                            std::cout << "alt: " << event.key.alt << std::endl;
                            std::cout << "shift: " << event.key.shift << std::endl;
                            std::cout << "system: " << event.key.system << std::endl;
                            //std::cout << "description: " << sf::Keyboard::getDescription(event.key.scancode).toAnsiString() << std::endl;
                            //std::cout << "localize: " << sf::Keyboard::localize(event.key.scancode) << std::endl;
                            //std::cout << "delocalize: " << sf::Keyboard::delocalize(event.key.code) << std::endl;
                        }
                        i++;
                    }
                    break;

                    // key pressed
                case sf::Event::KeyReleased:
                    // if (event.key.scancode == sf::Keyboard::Scan::Escape)
                    for (sf::Keyboard::Scancode& scan : inputs)
                    {
                        if (event.key.scancode == scan)
                        {
                            NoteOff* note = new NoteOff();

                            note->start = player.GetPlayerTime();
                            note->channel = 0;
                            note->key = notes[i];// 60;

                            player.notesPerTrack.resize(1);
                            player.InsertEvent(0, note);
                            std::cout << "the escape key was released" << std::endl;
                            std::cout << "scancode: " << event.key.scancode << std::endl;
                            std::cout << "code: " << event.key.code << std::endl;
                            std::cout << "control: " << event.key.control << std::endl;
                            std::cout << "alt: " << event.key.alt << std::endl;
                            std::cout << "shift: " << event.key.shift << std::endl;
                            std::cout << "system: " << event.key.system << std::endl;
                            //std::cout << "description: " << sf::Keyboard::getDescription(event.key.scancode).toAnsiString() << std::endl;
                            //std::cout << "localize: " << sf::Keyboard::localize(event.key.scancode) << std::endl;
                            //std::cout << "delocalize: " << sf::Keyboard::delocalize(event.key.code) << std::endl;
                        }
                        i++;
                    }
                    break;

                // we don't process other types of events
                default:
                    break;
            }
        }
    }
}