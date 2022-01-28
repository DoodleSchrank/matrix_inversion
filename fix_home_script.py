from subprocess import Popen, PIPE, call

binaries = ["./gcc-offload", "./clang-offload", "./nvcc-offload", "./gcc-offload-acc", "./gcc-offload-dbl", "./clang-offload-dbl", "./nvcc-offload-dbl", "./gcc-offload-acc-dbl"]
#binaries = ["./clang-offload"]

#call(["python", "benchmark-desy.py", binary, "openmp-offload"]).

arguments = ["./gcc-offload-dbl",
             "./gcc-offload-acc-dbl",
             "./clang-offload-dbl",
             "./nvcc-offload-dbl",]

for arg in arguments:
    out = call(["python", "benchmark-desy.py", arg])