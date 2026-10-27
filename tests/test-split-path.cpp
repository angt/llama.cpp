#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

// regression tests for the llama_split_path / llama_split_prefix buffer contract:
// on success the full name is written and its length is returned, otherwise 0
int main(void) {
    {
        std::vector<char> buf(64, '\0');
        int32_t len = llama_split_path(buf.data(), buf.size(), "/models/ggml-model-q4_0", 1, 4);
        assert(std::string(buf.data()) == "/models/ggml-model-q4_0-00002-of-00004.gguf");
        assert(len == (int32_t) std::string(buf.data()).size());
    }

    {
        // exact fit: name + NUL must fit, one byte less must not
        const std::string prefix = "/models/ggml-model-q4_0";
        const int32_t need = (int32_t) (prefix.size() + 20 + 1);

        std::vector<char> buf(need, '\0');
        assert(llama_split_path(buf.data(), buf.size(), prefix.c_str(), 1, 4) == need - 1);

        buf.assign(need - 1, '\0');
        assert(llama_split_path(buf.data(), buf.size(), prefix.c_str(), 1, 4) == 0);
    }

    {
        std::vector<char> buf(64, '\0');
        int32_t len = llama_split_prefix(buf.data(), buf.size(), "/models/ggml-model-q4_0-00002-of-00004.gguf", 1, 4);
        assert(len == 23);
        assert(std::string(buf.data()) == "/models/ggml-model-q4_0");
    }

    {
        // the numbers must match the name
        std::vector<char> buf(64, '\0');
        assert(llama_split_prefix(buf.data(), buf.size(), "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) == 0);
    }

    {
        // regression: a truncated prefix must return 0, never the untruncated length
        const std::string path = std::string(300, 'a') + "-00001-of-00002.gguf";

        std::vector<char> small(64, '\0');
        assert(llama_split_prefix(small.data(), small.size(), path.c_str(), 0, 2) == 0);

        std::vector<char> exact(301, '\0');
        int32_t len = llama_split_prefix(exact.data(), exact.size(), path.c_str(), 0, 2);
        assert(len == 300);
        assert(std::string(exact.data(), len) == std::string(300, 'a'));
    }

    {
        // UTF-8 names: lengths are byte counts, multibyte characters must survive intact
        // "cafe" with an accent, CJK and an emoji, then padding to exceed 260 bytes
        const std::string prefix = "caf\xC3\xA9_\xE4\xBD\xA0\xE5\xA5\xBD_\xF0\x9F\x98\x80_" + std::string(280, 'x');

        std::vector<char> buf(prefix.size() + 32, '\0');
        int32_t len = llama_split_path(buf.data(), buf.size(), prefix.c_str(), 0, 2);
        assert(len == (int32_t) (prefix.size() + 20));
        const std::string full(buf.data(), len);
        assert(full.compare(0, prefix.size(), prefix) == 0);

        std::vector<char> pbuf(prefix.size() + 1, '\0');
        int32_t plen = llama_split_prefix(pbuf.data(), pbuf.size(), full.c_str(), 0, 2);
        assert(plen == (int32_t) prefix.size());
        assert(std::string(pbuf.data(), plen) == prefix);
    }

    printf("test-split-path: all tests OK\n");

    return 0;
}
