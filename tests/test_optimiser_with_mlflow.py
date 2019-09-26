# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause
"""Test mlbox.optimisation.optimiser with mlflow integration module."""
import pytest
import numpy as np
import shutil
import os.path
import os
import pandas as pd

from mlbox.optimisation.optimiser import Optimiser
from mlbox.preprocessing.drift_thresholder import Drift_thresholder
from mlbox.preprocessing.reader import Reader
from mlbox.optimisation import make_scorer


@pytest.fixture
def setup_for_mlflow():
    try:
        shutil.rmtree('./save')
    except:
        pass


def test_evaluate_and_optimise_classification_with_mlflow(setup_for_mlflow):
    """Test evaluate_and_optimise method of Optimiser class."""
    reader = Reader(sep=",")

    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv",
                                          "data_for_tests/test.csv"],
                                   target_name="Survived")
    drift_thresholder = Drift_thresholder()
    drift_thresholder = drift_thresholder.fit_transform(dict)

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring='accuracy', n_folds=3)
    assert len(record) == 1
    dict_error = dict.copy()
    dict_error["target"] = dict_error["target"].astype(str)
    with pytest.raises(ValueError):
        score = opt.evaluate(None, dict_error)

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring='accuracy', n_folds=3)
    assert len(record) == 1
    score = opt.evaluate(None, dict)
    assert 0. <= score <= 1.

    space = {'ne__numerical_strategy': {"search": "choice", "space": [0]},
             'ce__strategy': {"search": "choice",
                              "space": ["label_encoding"]},
             'fs__threshold': {"search": "uniform",
                               "space": [0.01, 0.3]},
             'est__max_depth': {"search": "choice",
                                "space": [3, 4, 5, 6, 7]},
             'est__n_estimators': {'search': 'choice',
                                   'space': [100, 200, 300, 400]}

             }

    best = opt.optimise(space, dict, 4)

    # test for existence of data stored by mflow
    assert os.path.isfile('./save/mlflow_tracking/0/meta.yaml')
    assert len(os.listdir('./save/mlflow_tracking/0')) == 5

    # create pandas dataframe containing mflow captured data
    hyp_df = opt.extract_optimise_results()
    assert isinstance(hyp_df, pd.DataFrame)
    assert hyp_df.shape == (4, 36)

    # save pandas dataframe
    hyp_df.to_csv('./save/mlflow_data.csv', index=False)
    assert os.path.isfile('./save/mlflow_data.csv')



