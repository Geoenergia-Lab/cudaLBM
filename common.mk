# common.mk - Shared CUDA/C++ flags
NVCXX = nvcc
CXX = g++

-include $(CUDALBM_INCLUDE_DIR)/hardware.info

# CUDA Compiler Flags
NVCXX_STANDARD = -std c++20
NVCXX_OPTFLAGS = -O3 --restrict -extra-device-vectorization --maxrregcount=128
NVCXX_MFLAGS = --m64
NVCXX_WFLAGS = --Werror cross-execution-space-call,reorder,default-stream-launch,missing-launch-bounds,ext-lambda-captures-this -Xptxas -v
NVCXX_COMPILER_WFLAGS = -Xcompiler "-O3 -funroll-loops -march=native -mtune=native -Wall -Wextra -Werror -Wattributes -Wbuiltin-macro-redefined -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion -Wfloat-equal -Wformat-security -Wformat=2 -Wimplicit-fallthrough=5 -Winline -Wint-to-pointer-cast -Wlogical-op -Woverflow -Wpointer-arith -Wshadow -Wsign-conversion -Wstrict-aliasing=3 -Wstringop-overflow=4 -Wwrite-strings"
NVCXX_DFLAGS = -DSCALAR_PRECISION_32 -DLABEL_SIZE_64
NVCXX_FLAGS = $(NVCXX_STANDARD) $(NVCXX_OPTFLAGS) $(NVCXX_MFLAGS) $(NVCXX_ALL_ARCHFLAGS) $(NVCXX_WFLAGS) $(NVCXX_COMPILER_WFLAGS) $(NVCXX_DFLAGS)