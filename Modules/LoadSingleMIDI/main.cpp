#include <filesystem>
#include <iostream>
#include <fstream>
#include "MIDIParserException.h"
#include "LoggingMIDIParser.h"


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
    std::cout << "Please enter path : " << std::endl;

    std::string path;
    std::cin >> path;

    try 
    {
        LoggingMIDIParser parser("output.txt");

        std::ifstream file (path, std::ios::in|std::ios::binary|std::ios::ate);
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
            throw std::runtime_error("Couldn't open file : " + path);
        }

        displaySuccess("Loaded with success! " + path);
        // std::cout << "Loaded with success! " << path << std::endl;

        // std::ios::app is the open mode "append" meaning
        // new data will be written to the end of the file.
        std::ofstream out;
        out.open("config.txt", std::ios::app);
        out << path << " : " << "Success\n";
    }
    catch (const MIDIParserException& e)
    {
        displayError("MIDIParserException : " + std::string(e.what()));
        // std::cout << "MIDIParserException : " << e.what() << std::endl;

        // std::ios::app is the open mode "append" meaning
        // new data will be written to the end of the file.
        std::ofstream out;
        out.open("config.txt", std::ios::app);
        out << path << " : " << "Failure\n";
    }
    catch (const std::exception& e)
    {
        displayError("std::exception : " + std::string(e.what()));
        // std::cout << "std::exception : " << e.what() << std::endl;

        // std::ios::app is the open mode "append" meaning
        // new data will be written to the end of the file.
        std::ofstream out;
        out.open("config.txt", std::ios::app);
        out << path << " : " << "Failure\n";
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