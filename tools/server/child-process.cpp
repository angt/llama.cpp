#include "child-process.h"

#ifdef _WIN32
// ─── Windows implementation ─────────────────────────────────────────────────

#include <winsock2.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>

static std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return {};
    const int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (len == 0) return {};
    std::wstring ws(len - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, ws.data(), len);
    return ws;
}

// Build a Windows command-line string from argv, escaping as needed.
static std::string escape_cmdline(const std::vector<std::string> & args) {
    std::string result;
    for (size_t i = 0; i < args.size(); i++) {
        if (i > 0) result += ' ';

        const std::string & arg = args[i];
        bool needs_quote = arg.empty() ||
            arg.find_first_of(" \t\"") != std::string::npos;
        if (!needs_quote) {
            result += arg;
            continue;
        }

        result += '"';
        for (size_t j = 0; j < arg.size(); j++) {
            unsigned num_backslash = 0;
            while (j < arg.size() && arg[j] == '\\') {
                num_backslash++;
                j++;
            }
            if (j == arg.size()) {
                result.append(num_backslash * 2, '\\');
                break;
            } else if (arg[j] == '"') {
                result.append(num_backslash * 2 + 1, '\\');
                result += '"';
            } else {
                result.append(num_backslash, '\\');
                j--;
            }
        }
        result += '"';
    }
    return result;
}

// Build environment block from vector of "KEY=VALUE" strings.
static std::wstring build_env_block(const std::vector<std::string> & env) {
    std::wstring block;
    for (const auto & e : env) {
        block += utf8_to_wide(e);
        block += L'\0';
    }
    block += L'\0'; // double null terminator
    return block;
}

// ─── Windows method implementations ─────────────────────────────────────────

child_process::~child_process() {
    terminate();
    join();
}

child_process::child_process(child_process && o) noexcept
    : hProcess_(o.hProcess_)
    , stdin_file_(o.stdin_file_)
    , stdout_file_(o.stdout_file_)
{
    o.hProcess_    = nullptr;
    o.stdin_file_  = nullptr;
    o.stdout_file_ = nullptr;
}

child_process & child_process::operator=(child_process && o) noexcept {
    if (this != &o) {
        terminate();
        join();
        hProcess_    = o.hProcess_;
        stdin_file_  = o.stdin_file_;
        stdout_file_ = o.stdout_file_;
        o.hProcess_    = nullptr;
        o.stdin_file_  = nullptr;
        o.stdout_file_ = nullptr;
    }
    return *this;
}

int child_process::run(const std::vector<std::string> & argv,
                       const std::vector<std::string> & env) {
    if (hProcess_) return -1; // already running

    std::string cmdline = escape_cmdline(argv);
    std::wstring wcmdline = utf8_to_wide(cmdline);

    // Use argv[0] as the application name if it contains a path separator,
    // otherwise pass NULL to let CreateProcessW search via PATH.
    std::wstring wexe;
    if (!argv.empty()) {
        if (argv[0].find('/') != std::string::npos || argv[0].find('\\') != std::string::npos) {
            wexe = utf8_to_wide(argv[0]);
        }
    }

    // Create pipes for stdin/stdout
    SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};

    HANDLE stdin_read, stdin_write;
    if (!CreatePipe(&stdin_read, &stdin_write, &sa, 0)) return -1;
    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);

    HANDLE stdout_read, stdout_write;
    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
        CloseHandle(stdin_read);
        CloseHandle(stdin_write);
        return -1;
    }
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si = {0};
    si.cb = sizeof(si);
    si.dwFlags   = STARTF_USESTDHANDLES;
    si.hStdInput  = stdin_read;
    si.hStdOutput = stdout_write;
    si.hStdError  = stdout_write;

    PROCESS_INFORMATION pi = {0};

    DWORD flags = CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT;
    std::wstring env_block;
    LPCWSTR env_ptr = nullptr;
    if (!env.empty()) {
        env_block = build_env_block(env);
        env_ptr = env_block.data();
    }

    BOOL ok = CreateProcessW(
        wexe.empty() ? nullptr : wexe.c_str(),
        wcmdline.data(),
        nullptr, nullptr,
        TRUE, // inherit handles
        flags,
        (LPVOID)env_ptr,
        nullptr,
        &si, &pi
    );

    // Close the child-side handles (inherited by the child)
    CloseHandle(stdin_read);
    CloseHandle(stdout_write);

    if (!ok) {
        CloseHandle(stdin_write);
        CloseHandle(stdout_read);
        return -1;
    }

    // Close the thread handle immediately — we don't need it
    CloseHandle(pi.hThread);

    // Open C FILEs for the parent-side handles
    stdin_file_  = _fdopen(_open_osfhandle((intptr_t)stdin_write,  _O_WRONLY | _O_TEXT), "w");
    stdout_file_ = _fdopen(_open_osfhandle((intptr_t)stdout_read,  _O_RDONLY | _O_TEXT), "r");

    if (!stdin_file_ || !stdout_file_) {
        if (stdin_file_)  fclose(stdin_file_); else CloseHandle(stdin_write);
        if (stdout_file_) fclose(stdout_file_); else CloseHandle(stdout_read);
        stdin_file_  = nullptr;
        stdout_file_ = nullptr;
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        return -1;
    }

    hProcess_ = pi.hProcess;
    return 0;
}

bool child_process::is_alive() {
    if (!hProcess_) return false;
    DWORD exit_code = 0;
    if (!GetExitCodeProcess(hProcess_, &exit_code)) return false;
    return exit_code == STILL_ACTIVE;
}

void child_process::terminate() {
    if (hProcess_) {
        TerminateProcess(hProcess_, 1);
    }
}

int child_process::join() {
    if (!hProcess_) return -1;

    // Close pipes so the child sees EOF on stdin
    close_stdin();
    close_stdout();

    WaitForSingleObject(hProcess_, INFINITE);
    DWORD exit_code = 0;
    GetExitCodeProcess(hProcess_, &exit_code);

    CloseHandle(hProcess_);
    hProcess_ = nullptr;

    return static_cast<int>(exit_code);
}

#else
// ─── POSIX implementation ────────────────────────────────────────────────────

#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <spawn.h>

extern char ** environ;

// ─── POSIX method implementations ─────────────────────────────────────────────

child_process::~child_process() {
    terminate();
    join();
}

child_process::child_process(child_process && o) noexcept
    : pid_(o.pid_)
    , joined_(o.joined_)
    , wait_status_(o.wait_status_)
    , stdin_file_(o.stdin_file_)
    , stdout_file_(o.stdout_file_)
{
    o.pid_         = 0;
    o.joined_      = false;
    o.wait_status_ = 0;
    o.stdin_file_  = nullptr;
    o.stdout_file_ = nullptr;
}

child_process & child_process::operator=(child_process && o) noexcept {
    if (this != &o) {
        terminate();
        join();
        pid_          = o.pid_;
        joined_       = o.joined_;
        wait_status_  = o.wait_status_;
        stdin_file_   = o.stdin_file_;
        stdout_file_  = o.stdout_file_;
        o.pid_         = 0;
        o.joined_      = false;
        o.wait_status_ = 0;
        o.stdin_file_  = nullptr;
        o.stdout_file_ = nullptr;
    }
    return *this;
}

int child_process::run(const std::vector<std::string> & argv,
                       const std::vector<std::string> & env) {
    if (argv.empty()) return -1;
    if (pid_ > 0 || joined_) return -1; // already used

    // Create pipes for stdin (parent→child) and stdout (child→parent)
    int stdin_pipe[2];   // [0]=read (child), [1]=write (parent)
    int stdout_pipe[2];  // [0]=read (parent), [1]=write (child)
    if (pipe(stdin_pipe) != 0) return -1;
    if (pipe(stdout_pipe) != 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        return -1;
    }

    // Build C-string arrays before spawning (no allocation in child)
    std::vector<char *> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto & s : argv) {
        cargv.push_back(const_cast<char *>(s.c_str()));
    }
    cargv.push_back(nullptr);

    std::vector<std::string> mutable_env;
    std::vector<char *> cenv;
    if (!env.empty()) {
        mutable_env = env;
        cenv.reserve(mutable_env.size() + 1);
        for (auto & s : mutable_env) {
            cenv.push_back(s.data());
        }
        cenv.push_back(nullptr);
    }

    // Set up file actions: dup pipes to stdin/stdout/stderr, then close originals
    posix_spawn_file_actions_t fa;
    if (posix_spawn_file_actions_init(&fa) != 0) {
        close(stdin_pipe[0]);  close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        return -1;
    }

    posix_spawn_file_actions_adddup2(&fa, stdin_pipe[0],  STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&fa, stdout_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&fa, stdout_pipe[1], STDERR_FILENO);

    // Close pipe FDs in the child (both original and parent-side ends)
    posix_spawn_file_actions_addclose(&fa, stdin_pipe[0]);
    posix_spawn_file_actions_addclose(&fa, stdout_pipe[1]);
    posix_spawn_file_actions_addclose(&fa, stdin_pipe[1]);
    posix_spawn_file_actions_addclose(&fa, stdout_pipe[0]);

    // Reset signal handling to defaults in the child (parent may ignore SIGPIPE etc.)
    posix_spawnattr_t attr;
    if (posix_spawnattr_init(&attr) != 0) {
        posix_spawn_file_actions_destroy(&fa);
        close(stdin_pipe[0]);  close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        return -1;
    }

    sigset_t default_sigs;
    sigemptyset(&default_sigs);
    sigaddset(&default_sigs, SIGINT);
    sigaddset(&default_sigs, SIGTERM);
#ifdef SIGPIPE
    sigaddset(&default_sigs, SIGPIPE);
#endif
    posix_spawnattr_setsigdefault(&attr, &default_sigs);

    // Clear any signal mask the parent may have set
    sigset_t mask;
    sigemptyset(&mask);
    posix_spawnattr_setsigmask(&attr, &mask);

    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK);

    // posix_spawnp searches PATH (like execvp)
    pid_t pid = 0;
    int ret = posix_spawnp(
        &pid,
        argv[0].c_str(),
        &fa,
        &attr,
        cargv.data(),
        env.empty() ? environ : cenv.data()
    );

    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&fa);

    // Close the child-side pipe ends in the parent (no longer needed)
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    if (ret != 0) {
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        return -1;
    }

    // Open C FILEs for the parent-side pipe ends
    stdin_file_  = fdopen(stdin_pipe[1], "w");
    stdout_file_ = fdopen(stdout_pipe[0], "r");

    if (!stdin_file_ || !stdout_file_) {
        if (stdin_file_)  fclose(stdin_file_); else close(stdin_pipe[1]);
        if (stdout_file_) fclose(stdout_file_); else close(stdout_pipe[0]);
        stdin_file_  = nullptr;
        stdout_file_ = nullptr;
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        return -1;
    }

    pid_ = pid;
    return 0;
}

bool child_process::is_alive() {
    if (pid_ <= 0 || joined_) return false;

    int status = 0;
    int ret = waitpid(pid_, &status, WNOHANG);
    if (ret == 0) return true;    // still running

    if (ret == pid_) {
        // Child exited — reap the zombie and cache the wait status.
        joined_      = true;
        wait_status_ = status;
        pid_         = 0;
    }
    // ret == -1 (ECHILD etc.): treat as dead
    return false;
}

void child_process::terminate() {
    if (pid_ > 0) {
        kill(pid_, SIGKILL);
    }
}

int child_process::join() {
    // Already reaped by is_alive() — decode cached status
    if (joined_) {
        close_stdin();
        close_stdout();
        if (WIFEXITED(wait_status_)) return WEXITSTATUS(wait_status_);
        if (WIFSIGNALED(wait_status_)) return 128 + WTERMSIG(wait_status_);
        return -1;
    }
    if (pid_ <= 0) return -1;

    // Close pipes so the child sees EOF on stdin
    close_stdin();
    close_stdout();

    int status = 0;
    waitpid(pid_, &status, 0);
    joined_      = true;
    wait_status_ = status;
    pid_         = 0;

    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

#endif

// ─── Shared (platform-independent) methods ────────────────────────────────────

void child_process::close_stdin() {
    if (stdin_file_) { fclose(stdin_file_); stdin_file_ = nullptr; }
}

void child_process::close_stdout() {
    if (stdout_file_) { fclose(stdout_file_); stdout_file_ = nullptr; }
}