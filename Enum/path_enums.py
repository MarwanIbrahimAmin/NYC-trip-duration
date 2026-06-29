from enum import Enum

class DataTier(str, Enum):
    SAMPLE = "split_sample"  
    FULL = "split"       

class DataFile(str, Enum):
    TRAIN = "train.csv"
    VAL = "val.csv"
    TEST = "test.csv"