Modified Experiments

- unzip 'digits_dataset'  into a data folder within fccl+
- Make a 'data_wk/' folder inside data/
- Make a 'data_tensorboard/' folder inside fccl+/

- To run the experiments, run one of the shell scripts in `run.sh` file
- You can modify it accordingly
- All the arguments for different settings are inside `best_args.py`
- To modify which models to use, look at the fl_digits case under main.py and add the models in order of which you want to use them, you can make the list as big as you want.
- If you are having path issues please look at `conf.py` and check if all the directories exist

> All the techniques are implemented under models/ folder in fccl+

![Alt text](image-1.png)

some techniques require pretraining, this can be set at that specific models file, i.e at models/fccl.py line 29 and line 149

The order of the datasets are in MNIST, USPS, SVHN, SYNTHETIC
