
#Import System Util
import sys, os, time, platform
# Torch
import torch
import numpy as np
# import OCT
from oct_library import OCTProcessing
from jtop import jtop
from torch2trt import torch2trt

model_path = '/media/jetson/CA33-10E2/256.pth'
oc_file = '/home/jetson/Documents/oct_embedded/dataset/hopkins/hc01_spectralis_macula_v1_s1_R.vol'

model = torch.load(model_path, map_location='cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.ones((1, 1, 256, 256)).cuda()
model_trt = torch2trt(model, [x])
OCT = OCTProcessing(oct_file=oc_file, torchmodel=model_trt, half=True, device=device)
OCT.fovea_forward(imgw=256, imgh=256)

file = open('jtop_test_result_gpu_half_si_es_half.log', 'w')
file.write(f"ITERATION, GPUTEMP, CPUTEMP, GPUPOW, CPUPOW, FPS \n")

times = 20
dictmes = {'ms': [], 'fps': [], 'power_cpu': [], 'power_gpu': [] , 'temp_cpu': [], 'temp_gpu': []}

for i in range(times):
	#TEMP, POWER, 
	with jtop() as jetson:
		OCT.fovea_forward(imgw=256, imgh=256)
		ms = OCT.ms		
		fps = OCT.FPS
		powercpu = jetson.power[1]['SYS CPU']['avg']
		powergpu =  jetson.power[1]['SYS GPU']['avg']
		tempcpu = jetson.temperature['BCPU'] 
		tempgpu = jetson.temperature['GPU']

		dictmes['ms'].append(ms)
		dictmes['fps'].append(fps)
		dictmes['power_cpu'].append(powercpu)
		dictmes['power_gpu'].append(powergpu)
		dictmes['temp_cpu'].append(tempcpu)
		dictmes['temp_gpu'].append(tempgpu)

		print(f"ITERATION: {i} | MS: {ms} FPS: {fps} CPU POWER: {powercpu} GPU POWER: {powergpu} CPU TEMP: {tempcpu} GPU TEMP: {tempgpu}")

#print(dictmes)
index = 0
for key, value in dictmes.items():
	std = np.std(value)
	avg = np.average(value)
	if index == 2 or index == 3:
		print(f"{key} AVG: {avg/1000} W | {key} STD: {std/1000} W")
	elif index == 4 or index == 5:
		print(f"{key} AVG: {avg} °C | {key} STD: {std} °C")
	else:
		print(f"{key} AVG: {avg} | {key} STD: {std}")
	index = index + 1 