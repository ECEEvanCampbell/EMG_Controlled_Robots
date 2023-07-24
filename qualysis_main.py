import time
import asyncio
import qtm_rt
import os
QUALISYS_IP = "192.168.50.50" 

def on_packet(packet):
    # print(f"Framenumber: {packet.framenumber}")
    header, markers = packet.get_6d()
    # print(f"Component info {header}")
    with open("Data/turtlebot.csv", "a") as tb:
        line = str(time.perf_counter()) + "," +\
                str(markers[0][0][0]) + "," + \
                str(markers[0][0][1]) + "," + \
                str(markers[0][0][2]) + "\n"
        tb.writelines(line)
    with open("Data/copter.csv", "a") as cp:
        line = str(time.perf_counter()) + "," +\
                str(markers[2][0][0]) + "," + \
                str(markers[2][0][1]) + "," + \
                str(markers[2][0][2]) + "\n"
        cp.writelines(line)


async def setup():
    connection = await qtm_rt.connect(QUALISYS_IP)
    if connection is None: 
        return
    await connection.stream_frames(components=["6d"], on_packet=on_packet)


def main(start_perf_counter):
    asyncio.ensure_future(setup())
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    # cleanup
    if os.path.exists("Data/copter.csv"):
        os.remove("Data/copter.csv")
    if os.path.exists("Data/turtlebot.csv"):
        os.remove("Data/turtlebot.csv")
    # begin logging
    main(time.perf_counter())