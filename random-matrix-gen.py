#!/usr/bin/python
import random
import sys

size = int(sys.argv[1])

file = open("randomMatrix_" + str(size) + ".txt", "w")

for y in range(size):
	line = ""
	for x in range(size):
		line += str(random.uniform(-1, 1)) + " "
	file.write(line + "\n")

file.close()
