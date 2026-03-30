class BaseConfig:
    def __init__(self):
        # Basic settings
        self.SEED = 42
        self.DEVICE = "cuda"
        self.GPU_ID = 0

        # Path settings
        self.RAW_DATA_PATH = "./data/raw"
        self.PROCESSED_DATA_PATH = "./data/processed"
        self.CHECKPOINT_PATH = "./checkpoints"
        self.RESULT_PATH = "./results"
        self.LOG_PATH = "./logs"