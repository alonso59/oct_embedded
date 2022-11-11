import psutil
import GPUtil
print("="*40, "GPU Details", "="*40)
gpus = GPUtil.getGPUs()
list_gpus = []
for gpu in gpus:
    gpu_temperature = f"{gpu.temperature} °C"
    gpu_power = f"{gpu.power} °C"

psutil.sensors_temperatures() 