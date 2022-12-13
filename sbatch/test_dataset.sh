#!/bin/bash

rm output/miniconcode_test_preprocessed.pickle
rm output/miniconcode_train_preprocessed.pickle

python dataset.py

diff output/miniconcode_test_preprocessed*
diff output/miniconcode_train_preprocessed*