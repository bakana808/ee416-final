# ================================================================================
# PROGRAM PARAMETERS
# ================================================================================

CLASSES = ("covid", "healthy", "other")

# path to covid images
DATASET_COVID_PATH = "Data/Covid"

# path to healthy images
DATASET_HEALTHY_PATH = "Data/Healthy"

# path to "other pnumonia" images
DATASET_OTHER_PATH = "Data/Others"

# path to output all images to
DATASET_OUTPUT = "Dataset"

# path to move all images to
DATASET_TRAIN = "Train"

# ================================================================================
# TRAINING PARAMETERS
# ================================================================================

# number of samples to process through the model at a time
BATCH_SIZE = 150

# number of times to train the model on the same dataset
# more epochs = longer processing
NUM_EPOCHS = 10

# ================================================================================
# OPTIMIZER PARAMETERS
# ================================================================================

# learning rate; the higher this number is, the faster the weights adjust when training
LR = 0.001

# epsilon; the term added to the denominator to improve numerical stability
EPS = 1e-8

# penalty
WEIGHT_DECAY = 0

# ================================================================================
# MODEL PARAMTERS
# ================================================================================

# if true, linear layers will learn an additive bias
NN_LINEAR_BIAS = True

# NOTE: each image contains roughly ~90,000 pixels = ~90,000 features

# number of features to output in the first linear layer
NN_LINEAR_1_FOUT = 1000

# number of features to input in the second linear layer
NN_LINEAR_2_FIN = 1000
