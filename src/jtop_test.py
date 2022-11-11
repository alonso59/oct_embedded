
#Import System Util
import sys, os, time, platform
# Torch
import torch
import numpy as np
# import OCT
from oct_library import OCTProcessing
from jtop import jtop


def jetsonStats():
	with jtop() as jetson:
		print(f"BOARD: {jetson.board['info']['machine']}")
		print(f"SOC HARDWARE: {jetson.board['hardware']['SOC']}")
		print(f"CPU MODEL: {jetson.cpu['CPU1']['model']}")
		print(f"TEMP GPU: {jetson.temperature['GPU']} °C")
		print(f"TEMP BCPU: {jetson.temperature['BCPU']} °C")
		print(f"TEMP THERMAL: {jetson.temperature['thermal']} °C")
		print(f"TEMP TDIODE: {jetson.temperature['Tdiode']} °C")
		print(f"POWER GENERAL CURRENT: {jetson.power[0]['cur']} mW")
		print(f"POWER GENERAL AVG: {jetson.power[0]['avg']} mW")
		print(f"POWER SYS SOC CURRENT: {jetson.power[1]['SYS SOC']['cur']} mW")
		print(f"POWER SYS SOC AVG: {jetson.power[1]['SYS SOC']['avg']} mW")
		print(f"POWER SYS GPU CURRENT: {jetson.power[1]['SYS GPU']['cur']} mW")
		print(f"POWER SYS GPU AVG: {jetson.power[1]['SYS GPU']['avg']} mW")
		print(f"POWER SYS CPU CURRENT: {jetson.power[1]['SYS CPU']['cur']} mW")
		print(f"POWER SYS CPU AVG: {jetson.power[1]['SYS CPU']['avg']} mW")
		print(f"POWER SYS DDR CURRENT: {jetson.power[1]['SYS DDR']['cur']} mW")
		print(f"POWER SYS DDR AVG: {jetson.power[1]['SYS DDR']['avg']} mW")

times = 100

model_path = '/media/jetson/CA33-10E2/256.pth'
oc_file = '/home/jetson/Documents/oct_embedded/dataset/hopkins/hc01_spectralis_macula_v1_s1_R.vol'

model = torch.load(model_path, map_location='cuda')
model = model.half()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OCT = OCTProcessing(oct_file=oc_file, torchmodel=model, half=True, device=device)
OCT.fovea_forward(imgw=256, imgh=256)

file = open('jtop_test_result_gpu_half.log', 'w')
file.write(f"ITERATION, GPUTEMP, CPUTEMP, GPUPOW, CPUPOW, FPS \n")

for i in range(times):
	#TEMP, POWER, 
	with jtop() as jetson:
		OCT.fovea_forward(imgw=256, imgh=256)
		tempgpu = jetson.temperature['GPU']
		tempcpu = jetson.temperature['BCPU'] 
		powergpu =  jetson.power[1]['SYS GPU']['avg']
		powercpu = jetson.power[1]['SYS CPU']['avg']
		fps = OCT.FPS
		print(f"ITERATION: {i} | GPUTEMP: {tempgpu} CPUTEMP: {tempcpu} GPUPOW: {powergpu} CPUPOW: {powercpu} FPS: {fps}")
		file.write(f"{i}, {tempgpu}, {tempcpu}, {powergpu}, {powercpu}, {fps} \n")

file.close()
