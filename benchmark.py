#!/usr/bin/python
from subprocess import call

size = 2

while size < 10000:
    call(["./gcc/main", str(size), str(4)])
    size = size * 2