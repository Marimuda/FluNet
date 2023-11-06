#!/bin/bash

#python ./src/flunet/train.py +trainer.fast_dev_run=True experiment=example.yaml
python ./src/flunet/train.py +trainer.limit_train_batches=0.05 +trainer.limit_val_batches=0.05 +trainer.limit_test_batches=0.05 +trainer.log_every_n_steps=2 experiment=example.yaml
