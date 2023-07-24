import libemg
from config import Config
import models
import torch
import pickle

config = Config()

USER_ID = Config().subjectID

def launch_training():
    # p = libemg.streamers.sifibridge_streamer()
    p = libemg.streamers.myo_streamer()
    odh = libemg.data_handler.OnlineDataHandler()
    odh.start_listening()
    # Launch `training` ui
    training_ui = libemg.screen_guided_training.ScreenGuidedTraining()
    training_ui.download_gestures([1,2,3,4,5], "images/") # ,6,7,8,21,22,23,24,25,26,36,37
    training_ui.launch_training(odh, 5, 5, "images/", "Data/subject" + str(USER_ID) +"/SGT/", 3)
    p.kill()

def setup_classifier():
    # Step 1: Parse offline training data
    dataset_folder = 'Data/subject' +str(config.subjectID) + '/SGT/'
    classes_values = ["0","1","2","3","4"]
    classes_regex = libemg.utils.make_regex(left_bound = "_C_", right_bound=".csv", values = classes_values)
    reps_values = ["0", "1", "2","3","4"]
    reps_regex = libemg.utils.make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }

    odh = libemg.data_handler.OfflineDataHandler()
    odh.get_data(folder_location=dataset_folder, filename_dic=dic, delimiter=",")
    train_windows, train_metadata = odh.parse_windows(config.window_length, config.window_increment)
    fe = libemg.feature_extractor.FeatureExtractor()
    feature_list = fe.get_feature_groups()[config.features]
    features = fe.extract_features(feature_list, train_windows)

    fe.visualize_feature_space(features, "PCA",train_metadata["classes"])

    features = torch.hstack([torch.tensor(features[key], dtype=torch.float32) for key in features.keys()])
    #targets = torch.vstack([torch.eye(5, dtype=torch.float32)[i] for i in train_metadata["classes"]])
    targets = torch.tensor(train_metadata["classes"], dtype=torch.long)
    offline_classifier = libemg.emg_classifier.EMGClassifier()
    if hasattr(models, config.model):
        mdl_handle = getattr(models, config.model)
    else:
        exit(1)
    mdl = mdl_handle(input_shape = len(feature_list)*config.num_channels)
    mdl.memory.experience_data = features
    mdl.memory.experience_targets = targets
    mdl.memory.memories_stored = features.shape[0]
    mdl.memory.shuffle()
    mdl.train(config.epochs, shuffle_every_epoch=True)
    mdl.update_live_model()
    offline_classifier.__setattr__("classifier", mdl)

    offline_classifier.__setattr__("velocity", True)
    th_min_dic = {}
    th_max_dic = {}
    for i in range(5):
        # #for myo
        # th_min_dic[i] = 20
        # th_max_dic[i] = 90
        # FOR BIOARMBAND
        th_min_dic[i] = 1.5e-4
        th_max_dic[i] = 7.5e-4
    offline_classifier.__setattr__("th_min_dic", th_min_dic)
    offline_classifier.__setattr__("th_max_dic", th_max_dic)

    with open(dataset_folder + 'mdl.pkl','wb') as handle:
        pickle.dump(offline_classifier, handle)

if __name__ == "__main__":
    # launch_training()

    setup_classifier()
