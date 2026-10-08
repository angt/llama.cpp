#include "arg.h"

int llama_server(int argc, char ** argv);

#ifdef _WIN32
int wmain(int argc, wchar_t ** wargv) {
    return common_args_run(argc, wargv, llama_server);
}
#else
int main(int argc, char ** argv) {
    return llama_server(argc, argv);
}
#endif
