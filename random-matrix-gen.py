#!/usr/bin/python
# helper script to generate square matrices of size <arg> with values between -1 and 1
import random
import sys

#size = int(sys.argv[1])


size = 2
random.seed(2021)

while size < 20000:
	file = open("randomMatrix_" + str(size) + ".txt", "w")
	for y in range(size):
		line = ""
		for x in range(size):
			line += str(random.uniform(-1, 1)) + " "
		file.write(line + "\n")
	size = size * 2
file.close()
