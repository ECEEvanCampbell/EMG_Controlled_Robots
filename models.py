from torch import nn, optim
from torch.nn import functional as F
import torch
import copy
import numpy as np
import time
from config import Config
import math
import pickle
from utils import Memory
import libemg
config = Config()
import sys
thismodule = sys.modules[__name__]




ADAPTATION_EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
VADE_BATCHSIZE = 500
CL_BATCH_SIZEE = 500
RELABELLING_TRIGGER = 1000
NCLUSTERS = 10
HIDDEN_DIM = 2
LINE_SEARCH = 0.98
DET = 1e-10
VISUALIZE=True
VARIATIONAL_INFERENCE = False


# PARAMETERS: 
MAV_SPEED_MIN = 1.5e-4
MAV_SPEED_MAX = 5.0e-4

# ## SIFI bioarmband TDPSD
# means = torch.tensor([-0.76,-0.76,-0.76,-0.76,-0.76,-0.76,-0.76,-0.76, \
#                       -0.73,-0.73,-0.73,-0.73,-0.73,-0.73,-0.73,-0.73, \
#                       -0.15,-0.15,-0.15,-0.15,-0.15,-0.15,-0.15,-0.15, \
#                       -0.77,-0.77,-0.77,-0.77,-0.77,-0.77,-0.77,-0.77, \
#                        0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, \
#                       -0.62,-0.62,-0.62,-0.62,-0.62,-0.62,-0.62,-0.62])
# stds = torch.Tensor([ 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, \
#                       0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, \
#                       0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, \
#                       0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, \
#                       0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, \
#                       0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47])
# sifi htd
# means = torch.tensor([ 5.7e-5, 5.7e-5, 5.7e-5, 5.7e-5, 5.7e-5, 5.7e-5, 5.7e-5, 5.7e-5, \
#                         56.1, 56.1, 56.1, 56.1, 56.1, 56.1, 56.1, 56.1, \
#                         96.5, 96.5, 96.5, 96.5, 96.5, 96.5, 96.5, 96.5,\
#                         1.9e-3, 1.9e-3, 1.9e-3, 1.9e-3, 1.9e-3, 1.9e-3, 1.9e-3, 1.9e-3])
# stds = torch.tensor([ 7e-5, 7e-5, 7e-5, 7e-5, 7e-5, 7e-5, 7e-5,7e-5,\
#                         12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, 12.7, \
#                         22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3, \
#                         2.0e-3, 2.0e-3, 2.0e-3, 2.0e-3, 2.0e-3, 2.0e-3, 2.0e-3, 2.0e-3])
means = torch.tensor([6,6,6,6,6,6,6,6,\
        2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,\
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,\
        35,  35,  35,  35,  35,  35,  35,  35])
stds  = torch.tensor([6,6,6,6,6,6,6,6,\
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, \
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
        3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5])

def set_up_classifier(odh, savedir):
        
        fe = libemg.feature_extractor.FeatureExtractor()
        if config.features in ["MAP","ZC"]:
            feature_list = [config.features]
        else:
            feature_list = fe.get_feature_groups()[config.features]
        offline_classifier = libemg.emg_classifier.EMGClassifier()
        if hasattr(thismodule, config.model):
            mdl_handle = getattr(thismodule, config.model)
        else:
            exit(1)
        if mdl_handle == thismodule.AbstractDecoder:
            mdl = mdl_handle(num_classes=5)
        else:
            mdl = mdl_handle(input_shape = len(feature_list)*config.num_channels)
        offline_classifier.__setattr__("classifier", mdl)
        # this is a initialization for speed. 
        # TODO: fix this
        offline_classifier.__setattr__("velocity", True)
        th_min_dic = {}
        th_max_dic = {}
        for i in range(5):
            th_min_dic[i] = MAV_SPEED_MIN
            th_max_dic[i] = MAV_SPEED_MAX
        offline_classifier.__setattr__("th_min_dic", th_min_dic)
        offline_classifier.__setattr__("th_max_dic", th_max_dic)
        if config.SGT:
            with open( 'Data/subject' +str(config.subjectID) + '/SGT/mdl.pkl','rb') as file:
                trained_mdl = pickle.load(file)
                trained_weights = trained_mdl.classifier.models["background"]["classifier"].state_dict()
            # set the live model equal to the background model of the trained_mdl
            offline_classifier.classifier.setup_model("live")
            offline_classifier.classifier.setup_model("background")
            offline_classifier.classifier.models["background"]["classifier"].load_state_dict(trained_weights)
            offline_classifier.classifier.models["live"]["classifier"].load_state_dict(trained_weights)
        

        classifier = libemg.emg_classifier.OnlineEMGClassifier(offline_classifier=offline_classifier,
                                                                    window_size=config.window_length,
                                                                    window_increment=config.window_increment,
                                                                    online_data_handler=odh,
                                                                    features=feature_list,
                                                                    save_dir = savedir,
                                                                    save_predictions=True,
                                                                    output_format=config.oc_output_format,
                                                                    std_out=True)
        classifier.run(block=False)
        return classifier


class MLP(nn.Module):
    def __init__(self, input_shape):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.models = {}

        self.setup_model("live")
        self.setup_model("background")

        self.overwrite_model("live", "background")
        self.send_model_to("live","cpu")
        self.send_model_to("background","cpu")
        
        
        self.setup_optimizers("background")
        
        self.memory = Memory()
        self.batch_size = BATCH_SIZE
        self.frames_saved = 0

    def setup_model(self, name="live"):
        self.models[name] = {}
        self.models[name]["device"] = "cpu" 
        # The encoder is the same shape as the MLP's encoder:
        self.models[name]["classifier"] = nn.Sequential(
            nn.Linear(self.input_shape, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.Softmax(dim=1)
        )
        
    def send_model_to(self, name, where):
        for key in self.models[name].keys():
            if hasattr(self.models[name][key],"eval"):
                self.models[name][key].to(where)

    def setup_optimizers(self, name="background"):
        # set optimizer
        self.optimizer_classifier = optim.Adam(self.models[name]["classifier"].parameters(), lr=LEARNING_RATE)
        if config.loss_function == "MSELoss":
            self.loss_function = nn.MSELoss()
        elif config.loss_function == "CrossEntropyLoss":
            self.loss_function = nn.CrossEntropyLoss()
    
    def overwrite_model(self, to_be_overwritten="live", overwrite_with="background"):
        self.models[to_be_overwritten] = copy.deepcopy(self.models[overwrite_with])

    def update_live_model(self):
        self.overwrite_model()
        self.send_model_to("live","cpu")
        

    def normalize(self, x):
        if len(x):
            return (x - means)/stds

    def forward(self,x,name="live"):
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        x.requires_grad=False
        x = self.normalize(x)
        return self.models[name]["classifier"](x)
    
    def forward_reconstruct(self,x,name="background"):
        # to get a peek at the higher dimensional space
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        x.requires_grad=False
        x = self.normalize(x)
        x = self.models[name]["classifier"][0](x)
        x = self.models[name]["classifier"][1](x)
        x = self.models[name]["classifier"][2](x)
        x = self.models[name]["classifier"][3](x)
        x = self.models[name]["classifier"][4](x)
        return x
        
    def predict(self, data):
        probs = self.predict_proba(data)
        return np.array([np.where(p==np.max(p))[0][0] for p in probs])

    def predict_proba(self,data):
        if type(data) == np.ndarray:
            data = torch.tensor(data, dtype=torch.float32)
        output = self.forward(data, "live")
        return output.detach().numpy()

    def adapt(self, memory):
        self.memory = memory
        if not config.SGT:
            self.train()


    def train(self, epochs=ADAPTATION_EPOCHS, shuffle_every_epoch=False):
        num_batch = len(self.memory) // self.batch_size
        t = time.time()
        losses = []
        for e in range(epochs):
            if shuffle_every_epoch:
                self.memory.shuffle()
            loss = []
            batch_start = 0
            if num_batch > 0:
                for b in range(num_batch):
                    batch_end = batch_start + self.batch_size
                    self.optimizer_classifier.zero_grad()
                    predictions = self.forward(self.memory.experience_data[batch_start:batch_end,:], name="background")
                    loss_value = self.loss_function(predictions, self.memory.experience_targets[batch_start:batch_end])
                    loss_value.backward()
                    self.optimizer_classifier.step()
                    loss.append(loss_value.item())
                    batch_start = batch_end
                losses.append(sum(loss)/len(loss))
                print(f"E {e}: loss: {losses[-1]:.2f}")
            print("-"*15)
        elapsed_time = time.time() - t
        print(f"Adaptation_time = {elapsed_time:.2f}s" )


class AbstractDecoder:
    def __init__(self, num_classes=5):
        # Get num_classes random angles between -pi and pi
        if config.vector_method == "random":
            self.class_locations = np.random.uniform(-math.pi, math.pi, num_classes)
            self.class_locations[2] = 100 # we shouldn't make a no motion vector here. 
            # Abstract decoders dont usually have NM vectors
        # unintuitive but usable
        elif config.vector_method == "unintuitive":
            self.class_locations = np.array([math.pi/4, 3*math.pi/4, 0, -math.pi/4, -3*math.pi/4])
            # closed fist kind of goes right
            # hand open (thumb up) kind of goes up
            # supination kind of goes down
            # pronation kind of goes left
            self.class_locations[2] = 100 # we shouldn't make a no motion vector here. 
            # Abstract decoders dont usually have NM vectors
        elif config.vector_method == "intuitive":
        # more intuitive - this was for myo -- not checked for sifi
            self.class_locations = np.array([math.pi/4, -3*math.pi/4, 0, -math.pi/4, 3*math.pi/4])
        print(f"ABSTRACT_DECODER LOCATIONS \n {self.class_locations}")

    def forward(self,x):
        det = 1e-10
        distance_to_locations = 1/(np.abs(x - self.class_locations)+det)
        probabilities = np.exp(distance_to_locations)/np.sum(np.exp(distance_to_locations))
        return probabilities

    def __call__(self, *args, **kwargs):
        result = self.forward(*args, **kwargs)
        return result

    def predict(self, x):
        # used by libemg
        if type(x) != np.ndarray:
            x = x.detach().cpu().numpy()
        y = self.forward(x)
        predictions = torch.argmax(y, dim=1)
        return predictions.cpu().detach().numpy()


    def predict_proba(self, x):
        # used by libemg
        if type(x) != np.ndarray:
            x = x.detach().cpu().numpy()
        probabilities = self.forward(x)
        return probabilities
    


