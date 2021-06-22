#!/usr/bin/python
from subprocess import call

size = 2

while size < 10000:
    call(["./nvcc/main", str(size)])
    size = size * 2