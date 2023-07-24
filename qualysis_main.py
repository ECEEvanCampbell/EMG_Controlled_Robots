import time
import asyncio
import qtm_rt
import os
import socket
from config import Config
import numpy as np
config = Config()
QUALISYS_IP = "192.168.50.50" 

class QualysisManager:
    def __init__(self):
        self.in_port      = 12340
        self.qualysis_port = 12347

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("localhost", self.in_port))

        self.sock.sendto("GAMEPLAY".encode(), ("localhost", self.qualysis_port))

        self.last_turtle_loc = np.zeros(3)
        self.last_copter_loc = np.zeros(3)

    def on_packet(self,packet):
        # print(f"Framenumber: {packet.framenumber}")
        header, markers = packet.get_6d()
        # print(f"Component info {header}")
        timer = time.time()
        with open(f"data/subject{config.subjectID}/turtlebot.csv", "a") as tb:
            line = str(timer) + "," +\
                    str(markers[0][0][0]) + "," + \
                    str(markers[0][0][1]) + "," + \
                    str(markers[0][0][2]) + "\n"
            tb.writelines(line)
        with open(f"data/subject{config.subjectID}/copter.csv", "a") as cp:
            line = str(timer) + "," +\
                    str(markers[1][0][0]) + "," + \
                    str(markers[1][0][1]) + "," + \
                    str(markers[1][0][2]) + "\n"
            cp.writelines(line)
        turtle_loc = np.array([markers[0][0][0],
                               markers[0][0][1],
                               markers[0][0][2]])
        copter_loc = np.array([markers[1][0][0],
                               markers[1][0][1],
                               markers[1][0][2]])
        cur_dist   = turtle_loc[0:2] - copter_loc[0:2]
        last_dist = self.last_turtle_loc[0:2] - self.last_copter_loc[0:2]
        if not np.isnan(cur_dist).any() and not np.isnan(last_dist).any():
            if np.sum(cur_dist**2) < np.sum(last_dist**2):
                context = "P "
            else:
                context = "N "
            context += str(timer) + " "

            if cur_dist[1] < 0:
                # we need to go positive x
                context += "RIGHT "
            else:
                context += "LEFT "

            if cur_dist[0] < 0:
                # we need to go positive y
                context += "DOWN "
            else:
                context += "UP "
            
            with open(f"data/subject{config.subjectID}/context.csv", "a") as ctx_file:
                ctx_file.writelines(context+"\n")
            self.sock.sendto(context.encode(), ("localhost", self.qualysis_port))
        self.last_turtle_loc = turtle_loc
        self.last_copter_loc =  copter_loc


async def setup():
    connection = await qtm_rt.connect(QUALISYS_IP)
    if connection is None: 
        return
    qm = QualysisManager()
    await connection.stream_frames(components=["6d"], on_packet=qm.on_packet)


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

# sudo -E python3 qualysis_main.py&drone.py&emg_script.py