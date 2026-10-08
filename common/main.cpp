#include "main.h"

#ifdef _WIN32

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <windows.h>

// convert a wide value to UTF-8. throw when the value has no UTF-8 form
static std::string wide_to_utf8(const wchar_t * str) {
    int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, str, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        throw std::invalid_argument("error: cannot decode an argument as UTF-8");
    }
    std::string res(size, '\0');
    (void) WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, str, -1, res.data(), size, nullptr, nullptr);
    res.pop_back(); // drop the terminating NUL
    return res;
}

int wmain(int argc, wchar_t ** wargv) {
    std::vector<std::string> buf;
    std::vector<char *> ptrs;

    try {
        buf.reserve(argc);
        for (int i = 0; i < argc; ++i) {
            buf.push_back(wide_to_utf8(wargv[i]));
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    ptrs.reserve(buf.size() + 1);
    for (auto & val : buf) {
        ptrs.push_back(val.data());
    }
    ptrs.push_back(nullptr);

    return llama_main(argc, ptrs.data());
}

#else

int main(int argc, char ** argv) {
    return llama_main(argc, argv);
}

#endif
