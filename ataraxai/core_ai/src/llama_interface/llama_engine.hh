#include <string>
#include <vector>


class LlamaEngine {
public:
    LlamaEngine(const std::string& model_path);
    std::string prompt(const std::string& input);
    std::vector<float> embed(const std::string& input);
    void reset();
    ~LlamaEngine();
};
