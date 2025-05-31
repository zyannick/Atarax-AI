#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>
#include <cstdlib>

#include <csignal>
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
#include <unistd.h>
#include <sys/types.h>
#endif

#include <boost/process.hpp>
#include <boost/process/v1/io.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/args.hpp>
#include <boost/process/v1/search_path.hpp>
#include <boost/process/v1/start_dir.hpp>
#include <iostream>

#include "core_ai/io_utils/directory_utils.hh"

namespace bp = boost::process::v1;

class LlamaServerManager
{
    std::string llama_server_path = "llama-server";
    std::string model_path;
    int port = 8080;
    std::string mode;
    int max_context_per_user = 4096;
    int nb_users = 1;
    std::string draft_model_path;
    std::vector<std::string> command_args;
    std::string log_stdout;
    std::string log_stderr;
    std::string pid_file = "llama-server.pid";
    std::string output_dir = std::string(std::getenv("ATARAXIA_OUTPUT_DIR")) + "/llama-server";
    std::string pid_file_path = output_dir + "/" + pid_file;

    LlamaServerManager() = default;

    LlamaServerManager(const std::string &llama_server_path_param,
                       const std::string &model_path_param,
                       int port_param,
                       const std::string &mode_param,
                       int nb_users_param,
                       const std::string &draft_model_path_param)
        : llama_server_path(llama_server_path_param),
          model_path(model_path_param),
          port(port_param),
          mode(mode_param),
          nb_users(nb_users_param),
          draft_model_path(draft_model_path_param)
    {

        auto now_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::stringstream timestamp;

        timestamp << std::put_time(std::localtime(&now_time_t), "%Y%m%d-%H%M%S");

        log_stdout = "llama-server-" + mode + "-" + timestamp.str() + "-out.log";
        log_stderr = "llama-server-" + mode + "-" + timestamp.str() + "-err.log";

        command_args.push_back(llama_server_path);
        if (mode == "normal")
        {
            command_args.insert(command_args.end(), {"-m", model_path,
                                                     "--port", std::to_string(port)});
        }
        else if (mode == "multi-users")
        {
            int max_context_all_users = nb_users * max_context_per_user;
            command_args.insert(command_args.end(), {"-m", model_path,
                                                     "--port", std::to_string(port),
                                                     "-c", std::to_string(max_context_all_users),
                                                     "-np", std::to_string(nb_users)});
        }
        else if (mode == "speculative-decoding")
        {
            command_args.insert(command_args.end(), {"-m", model_path,
                                                     "--draft-model", draft_model_path,
                                                     "--port", std::to_string(port)});
        }
        else if (mode == "reranking")
        {
            command_args.insert(command_args.end(), {"-m", model_path,
                                                     "--reranking",
                                                     "--port", std::to_string(port)});
        }
        else
        {
            throw std::invalid_argument("Invalid mode specified: " + mode);
        }
    }

    void start()
    {
        if (command_args.empty())
        {
            std::cerr << "Error: Command arguments are not set up." << std::endl;
            throw std::runtime_error("Command arguments empty before starting server.");
        }

        try
        {
            std::cout << "Launching Llama server with ...\n";
            std::cout << "  Command: ";
            for (const auto &arg : command_args)
                std::cout << arg << " ";
            std::cout << "\n  Logs: " << log_stdout << ", " << log_stderr << "\n";

            boost::filesystem::path executable = command_args[0];
            std::vector<std::string> args_for_process;
            if (command_args.size() > 1)
            {
                args_for_process.assign(command_args.begin() + 1, command_args.end());
            }

            bp::child server_process(
                executable,
                bp::args(args_for_process),
                bp::std_out > log_stdout,
                bp::std_err > log_stderr,
                bp::start_dir("."));

            server_process.detach();

            std::ofstream pidfile(pid_file_path);
            if (pidfile)
            {
                pidfile << server_process.id() << "\n";
                pidfile.close();
                std::cout << "PID " << server_process.id() << " saved to: " << pid_file_path << "\n";
            }
            else
            {
                std::cerr << "Warning: Could not open PID file for writing: " << pid_file_path << "\n";
            }

            std::this_thread::sleep_for(std::chrono::seconds(2));

            std::string health_url = "http://localhost:" + std::to_string(port) + "/health";
            std::string curl_cmd = "curl --silent --fail " + health_url + " > /dev/null";

            int health_status = std::system(curl_cmd.c_str());

            if (health_status == 0)
            {
                std::cout << "Llama server passed health check.\n";
            }
            else
            {
                std::cerr << "Llama server failed health check at: " << health_url
                          << " (curl command exit status: " << health_status << ")\n";
            }
        }
        catch (const bp::process_error &e)
        {
            std::cerr << "Boost.Process Error launching llama-server: " << e.what() << std::endl;
            throw;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error launching llama-server: " << e.what() << std::endl;
            throw;
        }
    }

    void stop()
    {
        try
        {
            std::ifstream pidfile(pid_file_path);
            if (!pidfile)
            {
                std::cerr << "PID file not found: " << pid_file_path << ". Cannot determine process to stop.\n";
                return;
            }

            pid_t pid_to_kill = 0;
            pidfile >> pid_to_kill;
            pidfile.close();

            if (pid_to_kill <= 0)
            {
                std::cerr << "Invalid PID (0 or negative) in file: " << pid_file_path << "\n";
                return;
            }

            std::cout << "Attempting to stop Llama server with PID: " << pid_to_kill << "\n";

            int result = 0;
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
            result = kill(pid_to_kill, SIGTERM);
#else
            std::cerr << "Warning: 'kill' function for stopping process PID " << pid_to_kill
                      << " is typically POSIX-specific. Termination may not work as expected on this OS." << std::endl;
            result = ::kill(static_cast<int>(pid_to_kill), SIGTERM);
#endif

            if (result == 0)
            {
                std::cout << "Successfully sent SIGTERM to PID: " << pid_to_kill << "\n";
                std::remove(pid_file.c_str());
            }
            else
            {
                std::cerr << "Failed to send SIGTERM to PID: " << pid_to_kill << ". 'kill' returned " << result << ". It might already be stopped or permissions are lacking.\n";
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
                perror("  kill error");
#endif
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error stopping llama-server: " << e.what() << std::endl;
        }
    }

    void change_port(int new_port)
    {

        if (new_port <= 0 || new_port > 65535)
        {
            throw std::invalid_argument("Invalid port number: " + std::to_string(new_port));
        }
        this->stop();
        port = new_port;
        command_args.clear();
        command_args.push_back(llama_server_path);
        if (mode == "normal")
        {
            command_args.insert(command_args.end(), {"-m", model_path,
                                                     "--port", std::to_string(port)});
        }
        else if (mode == "multi-users")
        {
            int max_context_all_users = nb_users * max_context_per_user;
            command_args.insert(command_args.end(), {"-m", model_path,
                                                     "--port", std::to_string(port),
                                                     "-c", std::to_string(max_context_all_users),
                                                     "-np", std::to_string(nb_users)});
        }
        else if (mode == "speculative-decoding")
        {
            command_args.insert(command_args.end(), {"-m", model_path,
                                                     "--draft-model", draft_model_path,
                                                     "--port", std::to_string(port)});
        }
        else if (mode == "reranking")
        {
            command_args.insert(command_args.end(), {"-m", model_path,
                                                     "--reranking",
                                                     "--port", std::to_string(port)});
        }
        else
        {
            throw std::invalid_argument("Invalid mode specified: " + mode);
        }
        std::cout << "Changing port to " << port << " and restarting server...\n";
        this->start();
    }

    void restart()
    {
        try
        {
            this->stop();
            this->start();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during restart: " << e.what() << std::endl;
            throw;
        }
    }

    ~LlamaServerManager()
    {
        try
        {
            this->stop();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during destructor cleanup: " << e.what() << std::endl;
        }
    }
};
