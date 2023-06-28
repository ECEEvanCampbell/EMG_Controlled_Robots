import asyncio
import qtm_rt
from multiprocessing import Process
from multiprocessing import set_start_method

# Make sure you've installed the Qualisys 
# https://github.com/qualisys/qualisys_python_sdk
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
        # closure
        def on_packet(packet):
            header, markers = packet.get_6d()
            print("Component info: {}".format(header))
            for marker in markers:
                print("\t", marker)
        await self.connection.stream_frames(components=["6d"], on_packet=on_packet)

    def start_new_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            try:
                # Create the connection in the child process instead
                connection = loop.run_until_complete(qtm_rt.connect(QUALISYS_IP))
                # Use the connection for the `stream_frames` method
                loop.run_until_complete(connection.stream_frames(components=["6d"], on_packet=self.on_packet))
            except Exception as e:
                print(e)
        #loop.close()

    # No longer a closure, so it can be pickled
    def on_packet(self, packet):
        header, markers = packet.get_6d()
        print("Component info: {}".format(header))
        for marker in markers:
            print("\t", marker)


async def main():
    cs = ContextServer()
    await cs.run(block=False)


if __name__ == "__main__":
    # Use 'spawn' method to start a fresh Python interpreter for each child process
    set_start_method('spawn')
    asyncio.run(main())