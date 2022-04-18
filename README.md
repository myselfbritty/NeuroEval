# NeuroEval

Representation learning using rank loss for robust neurosurgical skills evaluation.


## Installation

To install this repository and its dependent packages, run the following.

~~~
git clone https://github.com/myselfbritty/NeuroEval.git
cd NeuroEval
conda create --name NeuroEval # (optional, for making a conda environment)
pip install -r requirements.txt
~~~

The [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) dataset extracted features by [et al.](https://github.com/Finspire13/Towards-Unified-Surgical-Skill-Assessment) can be downloaded from [here](https://drive.google.com/drive/folders/1fTDE764HVOAxUlaxWWc9fe66qSGoPxDi?usp=sharing), while NETS can be downloaded from [here](https://nets-iitd.github.io).

## Instructions to run

To train and test a model, make necessary changes in a config file and run the following command.
~~~
python3 main.py --config <config path> --log_dir <path to save logs>
~~~

For example, to run the default setting 4-fold validation on JIGSAWS, run the following.
~~~
python3 main.py --config configs/JIGSAWS_rank2/Knot_Tying/JIGSAWS_TVPE_4FOLD_Knot_Tying_0.json --log_dir logs/JIGSAWS_rank2
~~~

Similarly, to run the default setting train-test split on NETS, run the following.
~~~
python3 main.py --config configs/NETS_rank2/NETS_train_test.json --log_dir logs/NETS_rank2
~~~

