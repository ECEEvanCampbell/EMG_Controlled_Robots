import logging
import sys
import time
from threading import Event
import socket
import keyboard
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

HOST = '127.0.0.1'
PORT = 12346
DEFAULT_HEIGHT = 0.5
BOX_LIMIT = 1.5
DRONENUMBER = '0D'#'1A'##

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((HOST, PORT))

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7' + DRONENUMBER)
#URI = 'radio://0/80/2M/E7E7E7E719'
deck_attached_event = Event()
logging.basicConfig(level=logging.ERROR)

position_estimate = [0, 0]

def log_pos_callback(timestamp, data, logconf):
    print(data)
    global position_estimate
    position_estimate[0] = data['stateEstimate.x']
    position_estimate[1] = data['stateEstimate.y']

def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

def move_emg(scf):

    i = 0
    with MotionCommander(scf, default_height = DEFAULT_HEIGHT) as mc:
        while True:
            data, _ = s.recvfrom(1024)
            data = str(data.decode('utf-8'))
            file_parts = data.split(" ")
            probs = [float(i) for i in file_parts[:5]]
            classifier = np.argmax(probs)

            vel = float(file_parts[5]) * 0.25
            if (classifier == 0):
                if position_estimate[0] < -BOX_LIMIT:
                    mc.stop()
                else:
                    mc.start_forward(velocity=vel)
                i = 0
            elif (classifier == 1):
                if position_estimate[0] > BOX_LIMIT:
                    mc.stop()
                else:
                    mc.start_back(velocity=vel)
                i = 0
            elif (classifier == 3):
                if position_estimate[1] > BOX_LIMIT:
                    mc.stop()
                else:
                    mc.start_left(velocity=vel)
                i = 0
            elif (classifier == 4):
                if position_estimate[1] < -BOX_LIMIT:
                    mc.stop()
                else:
                    mc.start_right(velocity=vel)
                i = 0
            else:
                mc.stop()
                i = 0
            if (keyboard.is_pressed('q')):
                exit()
            i = i + 1

def move6dof_emg(scf):

    i = 0
    with MotionCommander(scf, default_height = DEFAULT_HEIGHT) as mc:
        while True:
            data, _ = s.recvfrom(1024)
            data = str(data.decode('utf-8'))
            classifier = data[0]
            vel = float(data[2:5])
            if (classifier == '1'):
                mc.start_forward(velocity=vel)
                i = 0
            elif (classifier == '0'):
                mc.start_back(velocity=vel)
                i = 0
            elif (classifier == '6'):
                mc.start_left(velocity=vel)
                i = 0
            elif (classifier == '5'):
                mc.start_right(velocity=vel)
                i = 0
            elif (classifier == '4'):
                mc.start_up(velocity=vel)
                i = 0
            elif (classifier == '3'):
                mc.start_down(velocity=vel)
                i = 0
            else:
                mc.stop()
                i = 0
            if (keyboard.is_pressed('q')):
                exit()
            i = i + 1
if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        scf.cf.param.add_update_callback(group="deck", name="bcFlow2",
                                cb=param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        logconf.start()

        move_emg(scf)

        logconf.stop()
