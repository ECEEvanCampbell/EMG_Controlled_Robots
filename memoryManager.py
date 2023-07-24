import socket
import select
from utils import Memory
import numpy as np
import pickle
import torch
import random
import time
import traceback
import logging



def worker(in_port, unity_port, out_port, save_dir, negative_method):
    logging.basicConfig(filename=save_dir + "memorymanager.log",
                        filemode='w',
                        encoding="utf-8",
                        level=logging.INFO)
    # this is where we receive commands from the classifier
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))

    # this is where we receive context from unity
    unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_sock.bind(("localhost", unity_port))

    # this is where we send out commands to classifier
    # managers only own their input sockets
    # out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # out_sock.bind(("localhost", out_port))

    # initialize the memory
    memory = Memory()

    adapt_file_line = 0 # we don't use this anymore :)
    predictions_file = save_dir + "predictions.csv"
    predictions_file_line = 0

    start_time = time.perf_counter()

    num_written = 0
    total_samples_written = 0
    waiting_flag = 0

    while True:
        try:
            # see what we have available:
            ready_to_read, ready_to_write, in_error = \
                select.select([in_sock, unity_sock], [], [],0)
            for sock in ready_to_read:
                received_data, _ = sock.recvfrom(512)
                data = received_data.decode("utf-8")
                
                if sock == unity_sock:
                    id_found, adaptation_data, adaptation_labels, adaptation_direction, adaptation_type, adaptation_group = \
                        decode_unity(data, predictions_file_line, predictions_file, negative_method)
                    memory.add_memories(adaptation_data, adaptation_labels, adaptation_direction,adaptation_type,adaptation_group)
                    predictions_file_line += id_found
                    # print(memory.memories_stored)
                elif sock == in_sock or waiting_flag:
                    if data == "WAITING" or waiting_flag:
                        # write memory to file
                        waiting_flag = 1
                        if len(memory):# never write an empty memory
                            t1 = time.perf_counter()
                            memory.write(save_dir, num_written)
                            del_t = time.perf_counter() - t1
                            logging.info(f"MEMORYMANAGER: WROTE FILE: {num_written}\t, lines:{len(memory)}, \t WRITE TIME: {del_t:.2f}s")
                            num_written += 1
                            memory = Memory()
                            in_sock.sendto("WROTE".encode("utf-8"), ("localhost", out_port))
                        waiting_flag = 0
        except:
            logging.error("MEMORYMANAGER: "+traceback.format_exc())
    

def decode_unity(packet, predictions_file_line, predictions_file, negative_method):
    class_map = ["DOWN","UP","NONE","RIGHT","LEFT", "UNUSED"]
    message_parts = packet.split(" ")
    outcome = message_parts[0]
    context = [message_parts[2], message_parts[3]]
    if context[0] == "UNUSED":
        return 
    timestamp = float(message_parts[1])
    # find the data
    feature_file = np.loadtxt(predictions_file, delimiter=",", skiprows=predictions_file_line)
    if len(feature_file.shape) == 1:
        feature_file = np.expand_dims(feature_file, axis=0)
    idx = np.argmin(np.abs(feature_file[:,0]-timestamp))
    # print(idx)
    features = torch.tensor(feature_file[idx,3:])
    prediction = feature_file[idx, 1]
    # PC = feature_file[idx,2]
    expected_direction = [class_map.index(context[0]),class_map.index(context[1])]
    
    group = get_group(outcome, prediction, expected_direction)

    adaptation_labels    = []
    adaptation_data      = []
    adaptation_data.append(features)
    adaptation_direction = []
    adaptation_direction.append(expected_direction)
    adaptation_outcome   = []
    adaptation_outcome.append(outcome)
    adaptation_group     = []
    adaptation_group.append(group)

    one_hot_matrix = torch.eye(5)


    if outcome == "P":
        # when its positive context, we make the adaptation target completely w/ 
        adaptation_labels.append(one_hot_matrix[int(prediction),:])
    elif outcome == "N":
        if negative_method == "random":
            choice = random.choice(expected_direction)
            adaptation_labels.append(one_hot_matrix[choice,:])
        elif negative_method == "all":
            adaptation_labels.append(one_hot_matrix[expected_direction[0],:])
            adaptation_labels.append(one_hot_matrix[expected_direction[1],:])
            # we need another copy of this stuff for "all"
            adaptation_data.append(features)
            adaptation_direction.append(expected_direction)
            adaptation_outcome.append(outcome)
            adaptation_group.append(group)
        elif negative_method == "mixed":
            mixed_label = torch.zeros(5)
            for o in expected_direction:
                mixed_label += one_hot_matrix[o,:]/len(expected_direction)
            adaptation_labels.append(mixed_label)

    adaptation_labels = torch.vstack(adaptation_labels).type(torch.float32)
    adaptation_data = torch.vstack(adaptation_data).type(torch.float32)
    adaptation_group = np.array(adaptation_group)
    adaptation_direction = np.array(adaptation_direction)
    adaptation_type = np.array([outcome])
    return idx, adaptation_data, adaptation_labels, adaptation_direction, adaptation_type, adaptation_group

def get_group(outcome, predictions, context):
        positive_group_map = [4,0,8,6,2]
        negative_group_map = [[-1, -1, -1, 5,  3],
                            [-1, -1, -1, 7,  1],
                            [-1, -1, 8, -1, -1],
                            [ 5,  7,-1, -1, -1],
                            [ 3,  1,-1, -1, -1]]
        if outcome == "P":
            # we get the group of the prediction
            group = positive_group_map[int(predictions)]
        elif outcome == "N":
            group = negative_group_map[context[0]][context[1]]
        return group