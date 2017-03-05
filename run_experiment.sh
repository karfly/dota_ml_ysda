#!/bin/bash
experiment_ipynb=$1

if [ "$#" -eq 0 ]
then
    echo "Usage: "
    echo "./run_experiment.sh [IPYTHON_NOTEBOOK]"
    exit
fi

nohup jupyter nbconvert --to notebook --execute $experiment_ipynb --ExecutePreprocessor.timeout=-1 > log.txt &