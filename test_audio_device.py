from wingbeat_detect import *

wh = WingbeatSensorHandler(obj_detect=False)

wh.bufferlist = deque(maxlen=wh.buffersize)
wh.p = pyaudio.PyAudio()

device_index = int(wh.device)

try:
    wh.p.open(format=wh.format,
                channels=wh.channels,
                rate=wh.rate,
                input_device_index=dev,
                input=True,
                frames_per_buffer=wh.chunk)
except:
    print("failed to open py audio stream")