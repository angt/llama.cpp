#pragma once

// child_process — Spawn child processes with piped stdio
//
// Usage:
//   child_process proc;
//   if (proc.run(args) != 0) { error; }
//   // write to proc.stdin_pipe(), read from proc.stdout_pipe()
//   proc.terminate();          // optional: force-kill
//   int code = proc.join();    // wait for exit, close pipes
//   // destructor handles terminate+join if not already done
//
// argv[0] is the command: if it contains '/', it's treated as a path;
// otherwise it is searched via PATH (posix_spawnp on POSIX, CreateProcessW
// on Windows).
//
// The recommended way to detect child exit is to read stdout_pipe() until EOF
// (the kernel closes the pipe when the child exits). is_alive() can be used
// as a secondary check.

#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/types.h>  // pid_t
#endif

class child_process {
public:
    child_process() = default;
    ~child_process();

    // Non-copyable
    child_process(const child_process &) = delete;
    child_process & operator=(const child_process &) = delete;
    // Movable
    child_process(child_process && o) noexcept;
    child_process & operator=(child_process && o) noexcept;

    // Spawn a command. argv[0] is the command (PATH-searched if not a path).
    // If env is empty, the child inherits the parent's environment.
    // If env is non-empty, it replaces the environment entirely (vector of "KEY=VALUE").
    // Returns 0 on success.
    int run(const std::vector<std::string> & argv,
            const std::vector<std::string> & env = {});

    // True if the child process is still running.
    // On POSIX, may reap the zombie and cache the exit status;
    // join() will still return the correct exit code.
    bool is_alive();

    // Force-kill the child (SIGKILL on POSIX, TerminateProcess on Windows).
    void terminate();

    // Wait for the child to exit and return its exit code.
    // Closes both pipes. Idempotent: safe to call multiple times.
    // Returns -1 if the process was never started.
    int join();

    // Pipe accessors (nullptr if not running or already joined)
    FILE * stdin_pipe()  const { return stdin_file_; }
    FILE * stdout_pipe() const { return stdout_file_; }

    // Explicitly close individual pipes when needed (e.g., to signal EOF on stdin).
    // Safe to call multiple times. join() also closes pipes automatically.
    void close_stdin();
    void close_stdout();

private:
#ifdef _WIN32
    HANDLE hProcess_ = nullptr;
#else
    pid_t pid_        = 0;
    bool  joined_     = false;
    int   wait_status_ = 0;
#endif
    FILE * stdin_file_  = nullptr;
    FILE * stdout_file_ = nullptr;
};