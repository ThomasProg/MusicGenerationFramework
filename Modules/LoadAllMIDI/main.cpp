#include <filesystem>
#include <iostream>
#include <fstream>
#include "MIDIParserException.h"
#include "LoggingMIDIParser.h"
#include "ConfigParser.h"


void displayError(const std::string& s)
{
    std::cout << "\033[1;31m" << s << "\033[0m\n" << std::endl;    
}

void displaySuccess(const std::string& s)
{
    std::cout << "\033[1;32m" << s << "\033[0m\n" << std::endl;    
}

void TryLoadAllFiles()
{
    size_t nbFailures = 0;

    size_t i = 0;
    std::string path = "C:/Users/thoma/Downloads/archive";

    ConfigParser config("config.txt");

    for (const auto& entry : std::filesystem::recursive_directory_iterator(path))
    {
        if (entry.is_directory() || entry.path().extension().string() != ".mid")
            continue;

        auto it = std::find_if(config.parsedLines.begin(), config.parsedLines.end(), ([&entry](const std::pair<std::string, std::string>& pair)
        {
            return pair.first == entry.path().string() && pair.second == "Success";
        }));

        i++;

        // corrupted file / crash : 
        // Tears_in_Heaven.7.mid
        //if (i == 5596)
        //    continue;

        // Crash, but is valid : 
        //  7300 / C:/Users/thoma/Downloads/archive\Jackson_Michael\Ill_Be_There.mid

        // if (i < 151)
        //     continue;
        if (it != config.parsedLines.end())
            continue;
        
        std::cout << "Trying to load : " << i << " / " << entry.path().string() << std::endl;

        try 
        {
            
            LoggingMIDIParser parser("output.txt");

            std::ifstream file (entry.path(), std::ios::in|std::ios::binary|std::ios::ate);
            if (file.is_open())
            {
                size_t size = file.tellg();
                char* memblock = new char [size];
                file.seekg (0, std::ios::beg);
                file.read (memblock, size);
                file.close();

                parser.LoadFromBytes(memblock, size);
                // parser.OnLoadedFromFile(filename);

                delete[] memblock;
            }
            else
            {
                throw std::runtime_error("Couldn't open file : " + entry.path().string());
            }

            displaySuccess("Loaded with success! " + entry.path().string());
            // std::cout << "Loaded with success! " << entry.path().string() << std::endl;

            // std::ios::app is the open mode "append" meaning
            // new data will be written to the end of the file.
            std::ofstream out;
            out.open("config.txt", std::ios::app);
            config.parsedLines.emplace_back(entry.path().string(), "Success\n");
            // out << entry.path().string() << " : " << "Success\n";
        }
        catch (const MIDIParserException& e)
        {
            displayError("MIDIParserException : " + std::string(e.what()));
            // std::cout << "MIDIParserException : " << e.what() << std::endl;

            // std::ios::app is the open mode "append" meaning
            // new data will be written to the end of the file.
            std::ofstream out;
            out.open("config.txt", std::ios::app);
            config.parsedLines.emplace_back(entry.path().string(), "Failure\n");
            // out << entry.path().string() << " : " << "Failure\n";
        }
        catch (const std::exception& e)
        {
            displayError("std::exception : " + std::string(e.what()));
            // std::cout << "std::exception : " << e.what() << std::endl;

            // std::ios::app is the open mode "append" meaning
            // new data will be written to the end of the file.
            std::ofstream out;
            out.open("config.txt", std::ios::app);
            config.parsedLines.emplace_back(entry.path().string(), "Failure\n");
            // out << entry.path().string() << " : " << "Failure\n";
        }
    }
}

int main()
{
    try 
    {
        TryLoadAllFiles();
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}