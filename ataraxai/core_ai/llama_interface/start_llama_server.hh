#include <string>
#include <vector>

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include <string>
#include <vector>

#include <cstdlib>
#include <cstdio>
#include <iostream>

#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

#include "subprocess.hpp"

using namespace subprocess;

struct start_llama_server
{
    std::string llama_server_path = "llama-server"; 
    std::string model_path;
    int port = 8080;
    std::string mode;
    int max_context_per_user = 4096;
    int nb_users = 1;
    std::string draft_model_path;

    std::string command;

    start_llama_server() = default;

    start_llama_server(const std::string &llama_server_path, const std::string &model_path, int port, const std::string &mode, int nb_users, const std::string &draft_model_path)
        : llama_server_path(llama_server_path), model_path(model_path), port(port), mode(mode), nb_users(nb_users), draft_model_path(draft_model_path)
    {
        if (mode == "normal")
        {
            command = llama_server_path + " -m " + model_path + " --port " + std::to_string(port);
        }
        else if (mode == "multi-users")
        {
            int max_context_all_users = nb_users * max_context_per_user;
            command = llama_server_path + " -m " + model_path + "  --port " + std::to_string(port) + " -c " + std::to_string(max_context_all_users) + " -np " + std::to_string(nb_users);
        }
        else if (mode == "speculative-decoding")
        {
            command = llama_server_path + " -m " + model_path + " --draft-model " + draft_model_path + " --port " + std::to_string(port);
        }
        else if (mode == "reranking")
        {
            command = llama_server_path + " -m " + model_path + " --reranking --port " + std::to_string(port);
        }
        else
        {
            throw std::invalid_argument("Invalid mode specified. Use 'normal' or 'draft'.");
        }
    }


    void start()
    {
        try{
            std::cout << "Starting Llama server with command: " << command << std::endl;
            int ret = system(command.c_str());
            if (ret != 0)
            {
                throw std::runtime_error("Failed to start Llama server. Command returned non-zero exit code.");
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error starting Llama server: " << e.what() << std::endl;
            throw;
        }
    }

};
