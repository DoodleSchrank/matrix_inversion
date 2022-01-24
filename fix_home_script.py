from subprocess import Popen, PIPE, call

binaries = ["./gcc-offload", "./clang-offload", "./nvcc-offload", "./gcc-offload-acc", "./gcc-offload-dbl", "./clang-offload-dbl", "./nvcc-offload-dbl", "./gcc-offload-acc-dbl"]
#binaries = ["./clang-offload"]

#call(["python", "benchmark-desy.py", binary, "openmp-offload"]).

for binary in binaries:
    out = call(["python", "benchmark-desy.py", binary])