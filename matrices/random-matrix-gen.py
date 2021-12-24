#!/usr/bin/python
# helper script to generate square matrices
import random
import sys

#size = int(sys.argv[1])


size = 2
random.seed(2021)

while size < 20000:
	file = open("randomMatrix_normal_" + str(size) + ".txt", "w")
	for y in range(size):
		line = ""
		for x in range(size):
			line += str(random.uniform(-1, 1)) + " "
		file.write(line + "\n")
	size = size * 2
	file.close()


size = 2
while size < 20000:
	file = open("randomMatrix_natural_" + str(size) + ".txt", "w")
	for y in range(size):
		line = ""
		for x in range(size):
			line += str(random.randint(-100, 100)) + " "
		file.write(line + "\n")
	size = size * 2
	file.close()


size = 2
while size < 20000:
	file = open("randomMatrix_sparse_" + str(size) + ".txt", "w")
	for y in range(size):
		line = ""
		for x in range(size):
			if x == y:
				line += str(random.uniform(-1, 1)) + " "
			else:
				line += "0.0 "
		file.write(line + "\n")
	size = size * 2
	file.close()


size = 2
while size < 20000:
	file = open("randomMatrix_triangle_" + str(size) + ".txt", "w")
	for y in range(size):
		line = ""
		for x in range(y):
			line += "0.0 "
		for x in range(y, size):
			line += str(random.uniform(-1, 1)) + " "
		file.write(line + "\n")
	size = size * 2
	file.close()
