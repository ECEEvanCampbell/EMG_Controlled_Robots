class Config:
    def __init__(self):
        self.window_length = 40#350#
        self.window_increment = 5 #56#
        self.features = 'LS4'
        self.num_channels = 8
        self.adapt_time = 10
        # Change this for each participant:
        self.subjectID = 0

        self.oc_output_format = "probabilities"

        self.model            = "MLP"
        self.negative_method  = "mixed"
        self.loss_function    = "MSELoss"

        self.epochs = 150
        self.SGT    = False
        if self.SGT:
            self.loss_function = "CrossEntropyLoss"