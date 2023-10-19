#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <ranges>
#include <string_view>
#include <iostream>

class ConfigParser
{
public:
    std::vector<std::pair<std::string, std::string>> parsedLines;

    ConfigParser(const char* configPath)
    {
        std::ifstream file (configPath, std::ios::in);
        if (file.is_open())
        {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string strFile = buffer.str();

            file.close();

            std::string_view memBlockView = strFile;

            for (const auto line : memBlockView | std::views::split('\n'))
            {
                using std::operator""sv;
                auto values = line | std::views::split(" : "sv);


                auto it = values.begin();
                std::string s1 = std::string(std::string_view(*it));
                std::advance(it, 1);
                std::string s2 = std::string(std::string_view(*it));
                parsedLines.emplace_back(std::move(s1), std::move(s2));
            }
        }
        // else
        // {
        //     throw std::runtime_error("Couldn't open file : " + std::string(configPath));
        // }
    }
};