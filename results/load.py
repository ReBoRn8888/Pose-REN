import numpy as np
# 76375
with open("NEUCOM18_MSRA_Pose_REN.txt") as file:
	data = file.readline().split(" ")
	del data[-1]
	print(len(data))
	x = []
	y = []
	d = []
	for i in range(int(len(data)/3)):
		x.append(data[i * 3])
		y.append(data[i * 3 + 1])
		d.append(data[i * 3 + 2])
	print("x = {}\ny = {}\nd = {}".format(x,y,d))