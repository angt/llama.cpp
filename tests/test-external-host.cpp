// External-host test: the reusable libraries (ggml, llama, mtmd) must accept UTF-8
// filenames from a host application that does NOT use our manifest. This binary is
// built without the manifest, so its narrow APIs run in the system code page.

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#undef NDEBUG
#include <cassert>

int main() {
    // only meaningful under a legacy (non-UTF-8) active code page
    if (GetACP() == CP_UTF8) {
        printf("test-external-host: SKIP (system code page is UTF-8)\n");
        return 0;
    }

    // a UTF-8 filename with non-ASCII characters (accent, CJK, emoji)
    const char * utf8_name = "caf\xC3\xA9-\xE6\xB5\x8B\xE8\xAF\x95-\xF0\x9F\x98\x80.txt";

    // the library must open it as UTF-8, not in the legacy code page
    FILE * f = ggml_fopen(utf8_name, "wb");
    assert(f != nullptr);
    fwrite("hello", 1, 5, f);
    fclose(f);

    f = ggml_fopen(utf8_name, "rb");
    assert(f != nullptr);
    char buf[6] = {};
    size_t n = fread(buf, 1, 5, f);
    fclose(f);
    assert(n == 5 && memcmp(buf, "hello", 5) == 0);

    // the file must exist under its exact Unicode name (wide API, code-page independent)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8_name, -1, nullptr, 0);
    assert(wlen > 0);
    std::wstring wname(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8_name, -1, &wname[0], wlen);
    assert(GetFileAttributesW(wname.c_str()) != INVALID_FILE_ATTRIBUTES);

    // clean up through the library's UTF-8 path handling
    std::error_code ec;
    std::filesystem::remove(std::filesystem::u8path(utf8_name), ec);

    printf("test-external-host: all tests OK\n");
    return 0;
}
