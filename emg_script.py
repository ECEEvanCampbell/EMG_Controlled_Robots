from multiprocessing import Process
import libemg
from models import set_up_classifier
import pickle
import socket
import os
import time
import numpy as np
import torch
import memoryManager
import adaptManager
import random
from config import Config
config = Config()

SEED_VALUE = 0
TIMEOUT = 305
SAVE_DIR = "data/subject" + str(config.subjectID)+ "/game/"

def fix_random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

def setup_folders():
    loc = "data/subject" + str(config.subjectID) + "/game/"
    if not os.path.isdir(loc):
        os.makedirs(loc)

class GameController:
    def __init__(self):
        self.unity_port = 12347
        self.memory_port = 12348
        self.classifier_port = 12349

        setup_folders()

        # self.p = libemg.streamers.sifibridge_streamer(notch_on=True, notch_freq=50,
        #                                          emg_fir_on=True,
        #                                          emg_fir=[20,450])
        self.p = libemg.streamers.myo_streamer()
        self.odh = libemg.data_handler.OnlineDataHandler(emg_arr=True)
        self.odh.start_listening()

        self.classifier = set_up_classifier(self.odh, SAVE_DIR)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('localhost', self.unity_port))
        self.sock.sendto(bytes(str("READY"),"utf-8"), ("localhost",12346))

    def run(self):
        print("Starting gameplay process")

        started=False
        print("READY!")
        memoryProcess = Process(target=memoryManager.worker, daemon=True, args=(self.classifier_port, self.unity_port, self.memory_port, SAVE_DIR, config.negative_method))
        # adaptProcess  = Process(target=adaptManager.worker, daemon=True, args=(self.memory_port, self.classifier_port, SAVE_DIR, self.classifier))
        while not started:
            try:
                received_data, _ = self.sock.recvfrom(1024)
                data = received_data.decode("utf-8")
                if data == "GAMEPLAY":
                    global_timer = time.perf_counter()
                    started=True
                    continue
            except:
                continue
        # close socket so we can use in thread:
        self.sock.close()
        # start game loop
        global_timer = time.perf_counter()
        # memoryManager.worker(self.classifier_port, self.unity_port, self.memory_port, SAVE_DIR, config.negative_method)
        

        memoryProcess.start()
        # adaptProcess.start()

        adaptManager.worker(self.memory_port, self.classifier_port, SAVE_DIR, self.classifier)
        
        while True:
            if (time.perf_counter() - global_timer > TIMEOUT):
                self.clean_up()
                # because we are running memory process with a daemon it dies when the main process dies
                exit(1)

    
    def clean_up(self):
        with open(SAVE_DIR + 'classifier_memory.pkl', 'wb') as handle:
            pickle.dump(self.classifier.classifier.classifier.memory, handle)
        # import matplotlib.pyplot as plt
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=2)
        # transformed_data = pca.fit_transform(self.classifier.classifier.classifier.memory.experience_data)
        # plt.scatter(transformed_data[:,0], transformed_data[:,1])
        delattr(self.classifier.classifier.classifier, "memory")
        with open(SAVE_DIR + 'mdl.pkl','wb') as handle:
            pickle.dump(self.classifier.classifier, handle)
        self.odh.stop_listening()
        self.classifier.stop_running()

if __name__ == "__main__":
    fix_random_seed(SEED_VALUE, False)
    gc = GameController()
    gc.run()

