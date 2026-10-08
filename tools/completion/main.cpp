#include "arg.h"

int llama_completion(int argc, char ** argv);

#ifdef _WIN32
int wmain(int argc, wchar_t ** wargv) {
    return common_args_run(argc, wargv, llama_completion);
}
#else
int main(int argc, char ** argv) {
    return llama_completion(argc, argv);
}
#endif
