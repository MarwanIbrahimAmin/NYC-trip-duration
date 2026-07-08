'''
load data
splitting data
apply preprocessing pipeline
return new data 

'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Preprocessing.preprocessing import Preprocessing_Pipeline
from data.load_data import load_train_val_test
from Enum.Feature import Feature
from Enum.path_enums import DataTier


class Preparing():
    def __init__(self, tier: DataTier = DataTier.SAMPLE):
        self.train, self.val, self.test = load_train_val_test(tier)
        self.preprocessor = Preprocessing_Pipeline()

    def prepare_data(self):
        train_processed = self.preprocessor.fit_transform(self.train)
        val_processed = self.preprocessor.transform(self.val)
        test_processed = self.preprocessor.transform(self.test)

        y_train = train_processed[Feature.TRIP_DURATION.value]
        y_val = val_processed[Feature.TRIP_DURATION.value]
        y_test = test_processed[Feature.TRIP_DURATION.value] if Feature.TRIP_DURATION.value in self.test.columns else None

        X_train = train_processed.drop(columns=[Feature.TRIP_DURATION.value], errors='ignore')
        X_val = val_processed.drop(columns=[Feature.TRIP_DURATION.value], errors='ignore')
        X_test = test_processed.drop(columns=[Feature.TRIP_DURATION.value], errors='ignore')





        return X_train, y_train, X_val, y_val, X_test, y_test
    


