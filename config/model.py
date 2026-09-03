from .base import BaseConfig

class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Dataset settings
        self.DATASET = "ICEWS18"

        # Model hyperparameters
        self.EMBED_DIM = 128
        self.GNN_LAYERS = 3
        self.TRANSFORMER_LAYERS = 3
        self.TRANSFORMER_HEADS = 8
        self.DECAY_RATE = 0.08

        # Training hyperparameters
        self.EPOCHS = 10
        self.LEARNING_RATE = 1e-4
        self.BATCH_SIZE = 32
        self.ALPHA = 0.5
        self.BETA = 0.7

        # Inference hyperparameters
        self.BEAM_DEPTH = 4
        self.BEAM_WIDTH = 4