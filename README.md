# RecResNet
RecResNet: A Recurrent Residual CNN Architecture for Disparity  Map Enhancement 3DV 2018. If you use this code please cite our paper [ RecResNet: A Recurrent Residual CNN Architecture for Disparity Map
Enhancement ] (http://personal.stevens.edu/~kbatsos/RecResNet.pdf)

```
@inproceedings{batsos2018recresnet,
  title={RecResNet: A Recurrent Residual CNN Architecture for Disparity Map Enhancement},
  author={Batsos, Konstantinos and Mordohai, Philipos},
  booktitle={ In International Conference on 3D Vision (3DV) },
  year={2018}
}

```
#Python

The code is using python 2.7 and tensorflow version 1.6.0

# CPP

The code includes two helper functions in c++. To compile the c++ code you will need boost python. You can safely omit these functions and replace them with your own.

#Training

The parameters for the datasets are provided in JSON format in the params folder. Please replace the paths of the data to reflect your filesystem.  All other parameters can be found in main.py. To train the network simply issue:

```
python main.py --params ./params/(dataset).json

```

The code saves data and variables to visualize the training process in tensorboard. 

#Testing

As provided the code can be used to test the whole dataset specified in the corresponding JSON file. To test a trained model simply issue:

```
python main.py --params ./params/(dataset).json --model (model path to load) --mode test

```

# Computing occlusiong masks for the synthetic dataset

If you would like to compute the occlusion masks for the synthetic dataset you can run the fr_occ.py code. 



