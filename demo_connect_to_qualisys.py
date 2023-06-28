
# Make sure you've installed the Qualisys 
# https://github.com/qualisys/qualisys_python_sdk
QUALISYS_IP = "192.168.50.50"

from multiprocessing import Process
import qtm_rt
import asyncio

# this is the context sever
class ContextServer:
    def __init__(self):
        self.process = Process(target = self.grab_packets, daemon=True,)

    async def run(self, block=True):
        if block:
            await self.grab_packets()
        else:
            await self.process.start()
    
    async def connect(self):
        self.connection = await qtm_rt.connect(QUALISYS_IP)


    async def grab_packets(self):
        # closure
        def on_packet(packet):
            header, markers = packet.get_6d()
            print("Component info: {}".format(header))
            for marker in markers:
                print("\t", marker)
        await self.connection.stream_frames(components=["6d"], on_packet=on_packet)
    

async def main():
    cs = ContextServer()
    await cs.connect()
    await cs.run(block=False)  

if __name__ == "__main__":
    asyncio.ensure_future(main())
    asyncio.get_event_loop().run_forever()