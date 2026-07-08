import os
import sys
import pytest
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from data.load_data import load_train_val_test, get_data_path
from Enum.path_enums import DataTier, DataFile
from Enum.Feature import Feature



@pytest.mark.parametrize("tier, expected_folder", [
    (DataTier.SAMPLE, "split_sample"),
    (DataTier.FULL, "split"),
])
@pytest.mark.parametrize("file_enum, expected_filename", [
    (DataFile.TRAIN, "train.csv"),
    (DataFile.VAL, "val.csv"),
    (DataFile.TEST, "test.csv"),
])
def test_get_data_path_all_combinations(tier, expected_folder, file_enum, expected_filename):
    path = get_data_path(tier=tier, filename=file_enum)
    
    assert expected_folder in path
    assert expected_filename in path
    assert os.path.isabs(path)



@pytest.mark.parametrize("tier", [DataTier.SAMPLE, DataTier.FULL])
def test_load_train_val_test_all_tiers(tier):
    train_df, val_df, test_df = load_train_val_test(tier=tier)

    assert isinstance(train_df, pd.DataFrame), f"train_df for {tier} is not a DataFrame"
    assert isinstance(val_df, pd.DataFrame), f"val_df for {tier} is not a DataFrame"
    assert isinstance(test_df, pd.DataFrame), f"test_df for {tier} is not a DataFrame"

    assert not train_df.empty, f"Training dataframe for {tier} is empty!"
    assert not val_df.empty, f"Validation dataframe for {tier} is empty!"
    assert not test_df.empty, f"Test dataframe for {tier} is empty!"

    target_col = Feature.TRIP_DURATION.value
    assert target_col in train_df.columns, f"'{target_col}' missing in Training Set ({tier})"
    assert target_col in val_df.columns, f"'{target_col}' missing in Validation Set ({tier})"