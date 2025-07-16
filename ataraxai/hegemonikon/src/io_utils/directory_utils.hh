#pragma once
#include <iostream>
#include <string>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

bool create_directory(const std::string &dirPath)
{
    fs::path directoryPath(dirPath);

    if (fs::exists(directoryPath))
    {
        if (fs::is_directory(directoryPath))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    try
    {
        if (fs::create_directory(directoryPath))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    catch (const fs::filesystem_error &e)
    {
        return false;
    }
    catch (const std::exception &e)
    {
        return false;
    }
}
