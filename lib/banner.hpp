#include <string>

namespace simulizer {
    static const std::string BANNER = []() {
        std::string s = R"(
    ____  ___ __  __ _   _ _    ___ _____ _____ ____
   / ___||_ _|  \/  | | | | |  |_ _||__  / ____|  _ \
   \___ \ | || |\/| | | | | |   | |   / /|  _| | |_) |
    ___) || || |  | | |_| | |___| |  / /_| |___|  _ <
   |____/___||_|  |_|\___/|_____|___/____|_____|_| \_\
       visit http://localhost:8080 for your results
)";
        s.erase(0, 1); // remove the first newline
        s.erase(s.find_last_not_of("\n") + 1); // remove trailing newlines
        return s;
    }();
}