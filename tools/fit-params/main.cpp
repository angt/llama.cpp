#include "arg.h"

int llama_fit_params(int argc, char ** argv);

#ifdef _WIN32
int wmain(int argc, wchar_t ** wargv) {
    return common_args_run(argc, wargv, llama_fit_params);
}
#else
int main(int argc, char ** argv) {
    return llama_fit_params(argc, argv);
}
#endif
