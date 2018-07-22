import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
import util
from util import get_center_fast as get_center
import config
from model_pose_ren import ModelPoseREN
import numpy as np
import cv2

from functools import partial
from net_deploy_baseline import make_baseline_net
from net_deploy_pose_ren import make_pose_ren_net

def print_usage():
	print('usage: {} icvl/nyu model_prefix out_file base_dir'.format(sys.argv[0]))
	exit(-1)

def show_results(img, results, dataset):
	img = np.minimum(img, 1500)
	img = (img - img.min()) / (img.max() - img.min())
	img = np.uint8(img*255)
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	img_show = util.draw_pose(dataset, img, results)
	return img_show

def MaxMinNormalization(x,Max,Min):
	x = (x - Min) / (Max - Min) * 255;
	return int(x);

def main():
	if len(sys.argv) < 3:
		print_usage()

	dataset = sys.argv[1]
	out_file = sys.argv[2]
	# data_dir_dict = {'nyu': config.nyu_data_dir,
	#                  'icvl': config.icvl_data_dir + "test/Depth",
	#                  'msra': config.msra_data_dir}
	# base_dir = data_dir_dict[dataset] #sys.argv[3]
	# name = os.path.join(base_dir, names[0])
	batch_size = 64
	if len(sys.argv) == 4:
		batch_size = int(sys.argv[3])

	# generate deploy prototxt
	make_baseline_net(os.path.join(ROOT_DIR, '../models'), dataset)
	make_pose_ren_net(os.path.join(ROOT_DIR, '../models'), dataset)


	names = util.load_names(dataset)
	# centers = util.load_centers(dataset)
	centers = None
	fx, fy, ux, uy = 587.270, 587.270, 326.548, 230.419
	# fx, fy, ux, uy = util.get_param(dataset)
	lower_ = 1
	upper_ = 650
	hand_model = ModelPoseREN(dataset,
		lambda img: get_center(img, lower=lower_, upper=upper_),
		param=(fx, fy, ux, uy), use_gpu=True)

	if dataset == 'msra':
		hand_model.reset_model(dataset, test_id = 0)

	base_dir = "/media/reborn/Others/Study/Reborn/Github/Pose-REN/test"
	# depthName = os.path.join(base_dir, "000000_depth.bin")
	depthName = os.path.join(base_dir, "0.img")
	imgName = os.path.join(base_dir, "image_0000.png")

	if(dataset == "msra"):
		# with open(depthName,"rb") as file:
		# 	data = np.fromfile(file, dtype=np.uint32)
		# 	width, height, left, top, right , bottom = data[:6]
		# 	depth = np.zeros((height, width), dtype=np.float32)
		# 	file.seek(4*6)
		# 	data = np.fromfile(file, dtype=np.float32)
		# depth[top:bottom, left:right] = np.reshape(data, (bottom-top, right-left))
		# depth[depth == 0] = 10000
		# print(depth[depth < depth.max()])
		# cv2.imwrite("img0.jpg",depth)

		with open(depthName,"rb") as file:
			data = np.fromfile(file, dtype=np.uint16)
			height = 480
			width = 640
			# depth = np.zeros((height, width), dtype=np.uint16)
			depth = np.reshape(data,(height,width)).astype(np.float32)
			min = depth.min()
			max = depth.max()
			print("min = {}, max = {}".format(min,max))

			flag = np.logical_xor(depth <= upper_, depth >= lower_)
			depth[flag] = 0

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))  
			depth = cv2.morphologyEx(depth,cv2.MORPH_CLOSE,kernel) 
			depth[depth == 0] = 10000
			# depth[depth == 0] = depth.max()

		# with open(depthName,"rb") as file:
		# 	depth = []
		# 	cnt = 0
		# 	contents = iter(partial(file.read, 2), b'')
		# 	for r in contents:
		# 		r_int = int.from_bytes(r, byteorder='big')  #将 byte转化为 int
		# 		cnt += 1
		# 		depth.append(r_int)
		# 	# print("i = {} -- {}".format(cnt,r_int))
		# depth = np.array(depth)
		# depth = np.reshape(depth,(480,640))
		# depth[depth == 0] = 10000
	elif(dataset == "icvl"):
		depth = cv2.imread(imgName, 2)
		depth[depth == 0] = depth.max()  # invalid pixel
		depth = depth.astype(float)
		



	# depth = np.reshape(depth,(240,320))
	# depth[depth == 0] = depth.max()
	print("np.shape(depth) = {}".format(np.shape(depth)))
	# depth = depth[:, ::-1]
	# print("names = {}".format(imgName))
	# print("np.shape(img) = {}".format(np.shape(img)))
	# results = hand_model.detect_files(base_dir, names, centers, max_batch=batch_size)
	results = hand_model.detect_image(depth)
	print("results = {}".format(results))
	print(np.shape(results))
	img_show = show_results(depth, results, dataset)
	cv2.imwrite('result.jpg', img_show)
	cv2.waitKey()
	# cv2.imshow('result', img_show)
	# util.save_results(results, out_file)

if __name__ == '__main__':
	main()

