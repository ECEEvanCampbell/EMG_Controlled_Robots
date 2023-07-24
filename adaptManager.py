import socket
import select
from utils import Memory
import numpy as np
import pickle
import time
import logging
import traceback
from config import Config
config = Config()

def worker(in_port, out_port, save_dir, online_classifier):
    logging.basicConfig(filename=save_dir + "adaptmanager.log",
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)
    # this is where we receive commands from the memoryManager
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))

    # this is where we write commands to the memoryManger
    # managers only own their input sockets
    # out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # out_sock.bind(("localhost", out_port))

    # initialize the memomry
    memory = Memory()
    memory_id = 0

    # initial time
    start_time = time.perf_counter()

    # variables to save and stuff
    adapt_round = 0

    time.sleep(1)
    
    while time.perf_counter() - start_time < 300:
        try:
            # see what we have available:
            ready_to_read, ready_to_write, in_error = \
                select.select([in_sock], [], [],0)
            # if we have a message on the in_sock get the message
            # this means we have new data to load in from the memory manager
            for sock in ready_to_read:
                received_data, _ = sock.recvfrom(1024)
                data = received_data.decode("utf-8")
                # we were signalled we have data we to load
                if data == "WROTE":
                    # append this data to our memory
                    t1 = time.perf_counter()
                    new_memory = Memory()
                    new_memory.from_file(save_dir, memory_id)
                    memory += new_memory
                    del_t = time.perf_counter() - t1
                    memory_id += 1
                    logging.info(f"ADAPTMANAGER: ADDED MEMORIES, \tCURRENT SIZE: {len(memory)}; \tLOAD TIME: {del_t:.2f}s")
            # if we still have no memories (rare edge case)
            if not len(memory):
                logging.info("NO MEMORIES -- SKIPPED TRAINING")
            else:
                t1 = time.perf_counter()
                online_classifier.classifier.classifier.adapt(memory)
                online_classifier.classifier.classifier.update_live_model()
                online_classifier.raw_data.set_classifier(online_classifier.classifier.classifier)
                del_t = time.perf_counter() - t1
                logging.info(f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s")
                adapt_round += 1
                time.sleep(2)
            
            # signal to the memory manager we are idle and waiting for data
            in_sock.sendto("WAITING".encode("utf-8"), ("localhost", out_port))
            logging.info("ADAPTMANAGER: WAITING FOR DATA")
            time.sleep(0.5)
        except:
            logging.error("ADAPTMANAGER: "+traceback.format_exc())
    else:
        memory.write(save_dir, 1000)


        


