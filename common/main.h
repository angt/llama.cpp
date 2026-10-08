#pragma once

// entry point of an executable, called by common/main.cpp with the arguments of the process
// the values of argv are UTF-8 on Windows and native bytes elsewhere
int llama_main(int argc, char ** argv);
