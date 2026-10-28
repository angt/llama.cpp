// Windows startup tests: UTF-8 process code page, exact argv bytes, embedded application manifest.
// The checks run before any locale or library init, so they see the raw process state.

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "common.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

// UTF-8 byte sequences, kept as escapes so this file stays ASCII
static const char * const CHILD_ARGS[] = {
    "caf\xC3\xA9",                                          // latin with an accent
    "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82",     // cyrillic
    "\xE4\xBD\xA0\xE5\xA5\xBD",                             // CJK
    "\xF0\x9F\x98\x80",                                     // emoji (non-BMP)
    "with space",
    "",                                                     // empty argument
    "quote\"inside",
    "trailing\\",                                           // trailing backslash
    "C:\\path\\to\\",
};

static std::wstring get_self_path() {
    std::vector<wchar_t> buf(MAX_PATH);
    while (true) {
        // GetModuleFileNameW returns the buffer size when the path does not fit
        DWORD len = GetModuleFileNameW(nullptr, buf.data(), (DWORD) buf.size());
        assert(len > 0);
        if (len < buf.size()) {
            return std::wstring(buf.data(), len);
        }
        buf.resize(buf.size() * 2);
    }
}

// the application manifest must be embedded as RT_MANIFEST ID 1 with both settings
static bool check_manifest(const wchar_t * exe_path) {
    HMODULE module = nullptr;
    if (exe_path) {
        module = LoadLibraryExW(exe_path, nullptr, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
        if (!module) {
            fprintf(stderr, "cannot load %ls\n", exe_path);
            return false;
        }
    }

    HRSRC res = FindResourceW(module, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(24)); // RT_MANIFEST
    if (!res) {
        fprintf(stderr, "no RT_MANIFEST ID 1 in %ls\n", exe_path ? exe_path : L"<self>");
        if (module) {
            FreeLibrary(module);
        }
        return false;
    }

    const char * data = (const char *) LockResource(LoadResource(module, res));
    const DWORD size = SizeofResource(module, res);
    const std::string xml(data, size);

    if (module) {
        FreeLibrary(module);
    }

    return xml.find(">UTF-8</activeCodePage>") != std::string::npos
        && xml.find(">true</longPathAware>") != std::string::npos;
}

// read the LongPathsEnabled machine policy: 1 on, 0 off (the default, also when the value is absent), -1 when the state cannot be read
static int query_long_paths_policy() {
    DWORD value = 0;
    DWORD size = sizeof(value);
    LSTATUS st = RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\FileSystem", L"LongPathsEnabled", RRF_RT_REG_DWORD, nullptr, &value, &size);
    if (st == ERROR_SUCCESS) {
        return value ? 1 : 0;
    }
    if (st == ERROR_FILE_NOT_FOUND || st == ERROR_PATH_NOT_FOUND) {
        return 0;
    }
    return -1;
}

// quote one argument the way the Windows CRT and CommandLineToArgvW parse it back
static std::wstring quote_arg(const std::wstring & arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return arg;
    }
    std::wstring out = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') {
            backslashes++;
        } else if (c == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out += L'"';
            backslashes = 0;
        } else {
            out.append(backslashes, L'\\');
            backslashes = 0;
            out += c;
        }
    }
    out.append(backslashes * 2, L'\\');
    out += L'"';
    return out;
}

// the child validates the argv bytes it received: they must be exact UTF-8
static int run_probe_child(int argc, char ** argv) {
    if (GetACP() != CP_UTF8) {
        fprintf(stderr, "GetACP() = %u, expected %u\n", GetACP(), (unsigned) CP_UTF8);
        return 1;
    }
    const int n_expected = (int) (sizeof(CHILD_ARGS) / sizeof(CHILD_ARGS[0]));
    if (argc != n_expected + 2) {
        fprintf(stderr, "argc = %d, expected %d\n", argc, n_expected + 2);
        return 1;
    }
    for (int i = 0; i < n_expected; i++) {
        if (strcmp(argv[i + 2], CHILD_ARGS[i]) != 0) {
            fprintf(stderr, "argv[%d] mismatch\n", i + 2);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    // probe the process state first, before any locale or library init
    assert(GetACP() == CP_UTF8);

    if (argc >= 2 && strcmp(argv[1], "--probe-child") == 0) {
        return run_probe_child(argc, argv);
    }

    assert(check_manifest(nullptr));

    // CTest passes the shipped executables here: each must carry the same manifest
    for (int i = 1; i < argc; i++) {
        assert(check_manifest(utf8_to_wstring(argv[i]).c_str()));
    }

    {
        // spawn this exe with a crafted command line: the child compares raw argv bytes
        std::wstring cmdline = quote_arg(get_self_path()) + L" --probe-child";
        for (const char * arg : CHILD_ARGS) {
            cmdline += L" " + quote_arg(utf8_to_wstring(arg));
        }

        std::vector<wchar_t> cmd_buf(cmdline.begin(), cmdline.end());
        cmd_buf.push_back(0);

        STARTUPINFOW si = {};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi = {};
        BOOL ok = CreateProcessW(get_self_path().c_str(), cmd_buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi);
        assert(ok);
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD code = 1;
        GetExitCodeProcess(pi.hProcess, &code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        assert(code == 0);
    }

    {
        // a path can exceed 260 UTF-8 bytes while staying under 260 UTF-16 units: no byte-length limits allowed
        std::error_code ec;
        std::filesystem::path dir = std::filesystem::temp_directory_path() / "llama-test-win-startup";
        const std::string comp = "\xE6\xA8\xA1\xE5\x9E\x8B\xE7\x9B\xAE\xE5\xBD\x95"; // CJK, 3 bytes per char in UTF-8
        // grow past 260 UTF-8 bytes, counting the temporary-directory prefix
        std::filesystem::path file;
        for (int i = 0; ; i++) {
            dir /= std::filesystem::path(comp + std::to_string(i));
            file = dir / "file.txt";
            if (file.string().size() > 260) {
                break;
            }
        }
        std::filesystem::create_directories(dir, ec);
        assert(!ec);

        assert(file.string().size() > 260);
        assert(file.wstring().size() < 260);

        {
            std::ofstream out(file);
            out << "hello";
        }
        assert(std::filesystem::exists(file));

        std::filesystem::remove_all(std::filesystem::temp_directory_path() / "llama-test-win-startup", ec);
    }

    {
        // absolute paths beyond MAX_PATH need LongPathsEnabled=1 on the machine: skip when the policy is off
        std::error_code ec;
        std::filesystem::path base = std::filesystem::temp_directory_path() / "llama-test-win-startup-long";
        // a short control: if this fails the error is not about path length, and must fail the test
        std::filesystem::create_directories(base, ec);
        assert(!ec);

        std::filesystem::path dir = base;
        for (int i = 0; i < 20; i++) {
            dir /= "component-with-a-long-name";
        }
        assert(dir.wstring().size() > 260); // genuinely long: this is the LongPathsEnabled case

        std::filesystem::create_directories(dir, ec);
        if (ec) {
            // skip only when the policy is known to be off: the short base does not prove the cause
            const int policy = query_long_paths_policy();
            if (policy != 0) {
                fprintf(stderr, "cannot create long directory: %s (LongPathsEnabled policy: %s)\n", ec.message().c_str(), policy < 0 ? "unknown" : "on");
            }
            assert(policy == 0);
            printf("test-win-startup: SKIP long path test (LongPathsEnabled=1 required): %s\n", ec.message().c_str());
        } else {
            std::filesystem::path file = dir / "file.txt";
            assert(file.string().size() > 260);

            std::ofstream out(file);
            out << "hello";
            out.close();
            assert(out.good());
            assert(std::filesystem::exists(file));
        }
        std::filesystem::remove_all(base, ec);
    }

    {
        // a UTF-8 environment value used as a path must round-trip through the app's env helper
        const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llama-test-win-startup-env-\xC3\xA9\xE6\xB5\x8B";
        common_set_env("LLAMA_TEST_WIN_STARTUP_PATH", dir.string());
        const std::filesystem::path env_dir = common_get_path_from_env("LLAMA_TEST_WIN_STARTUP_PATH");
        assert(env_dir == dir);
        std::error_code ec;
        std::filesystem::create_directories(env_dir, ec);
        assert(!ec);
        std::ofstream out(env_dir / "file.txt");
        out << "hello";
        out.close();
        assert(std::filesystem::exists(env_dir / "file.txt"));
        common_set_env("LLAMA_TEST_WIN_STARTUP_PATH", "");
        std::filesystem::remove_all(dir, ec);
    }

    printf("test-win-startup: all tests OK\n");

    return 0;
}
