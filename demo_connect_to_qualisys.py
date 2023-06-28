import asyncio
import qtm_rt
from multiprocessing import Process
from multiprocessing import set_start_method
import time
# Make sure you've installed the Qualisys 
# https://github.com/qualisys/qualisys_python_sdk
from multiprocessing import shared_memory
import numpy as np

QUALISYS_IP = "192.168.50.50"


# this is the context sever
class ContextServer:
    def __init__(self):
        self.process = None
        
    async def run(self, block=True):
        if block:
            await self.grab_packets()
        else:
            self.process = Process(target=self.start_new_loop)
            self.process.start()

    async def connect(self):
        self.connection = await qtm_rt.connect(QUALISYS_IP)

    async def grab_packets(self):
        await self.connect()
        # closure
        def on_packet(packet):
            header, markers = packet.get_6d()
            print("Component info: {}".format(header))
            existing_buffer = shared_memory.SharedMemory(name="context")
            location = np.ndarray((100,6), dtype=np.float32, buffer=existing_buffer.buf)
            
            new_location = np.array([markers[0][0][0],
                                     markers[0][0][1],
                                     markers[0][0][2],
                                     markers[1][0][0],
                                     markers[1][0][1],
                                     markers[1][0][2]],dtype=np.float32).reshape((1,6))
            location = np.concatenate((location[-99:,:], new_location),axis=0)

        await self.connection.stream_frames(components=["6d"], on_packet=on_packet)

    def start_new_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Create the connection in the child process instead
        connection = loop.run_until_complete(qtm_rt.connect(QUALISYS_IP))
        while True:
            try:
                # Use the connection for the `stream_frames` method
                loop.run_until_complete(connection.stream_frames(components=["6d"], on_packet=self.on_packet))
            except Exception as e:
                print(f"Error: {e}")
        loop.close()

    # No longer a closure, so it can be pickled
    def on_packet(self, packet):
        header, markers = packet.get_6d()
        print("Component info: {}".format(header))
        for marker in markers:
            print("\t", marker)


async def main():
    # prepare the shared memory buffer for locations
    # 100 samples of x,y,z ground, x,y,z 
    locations = np.zeros((100,6), dtype=np.float32)
    shared_memory_buffer = shared_memory.SharedMemory(create=True, size=locations.nbytes, name="context")
    locations = np.ndarray(locations.shape, dtype=np.float32, buffer=shared_memory_buffer.buf)
    # start context server
    cs = ContextServer()
    await cs.run(block=True)

    while True:
        time.sleep(1)
        print(locations[-1,:])


if __name__ == "__main__":

    # Use 'spawn' method to start a fresh Python interpreter for each child process
    set_start_method('spawn')
    asyncio.run(main())