from yacs.config import CfgNode as CN

_C = CN()
# ---------------------------------------------------------------- #
#         STATE REPRESENTATION LEARNING CONFIGURATIONS             #
# ---------------------------------------------------------------- #
_C.SRL = CN()
_C.SRL.USE = False
_C.SRL.MODELS = []
_C.SRL.LOSSES = []
_C.SRL.WEIGHTS = []
_C.SRL.UPDATE_EVERY= 100
_C.SRL.UPDATE_AFTER= 1000
_C.SRL.BATCH_SIZE = 128
_C.SRL.INITIAL_UPDATE_STEPS = 0
_C.SRL.ONLY_PRETRAINING = False
_C.SRL.JOINT_TRAINING = False

# ----------------> LATENT MODEL <--------------- #
_C.SRL.LATENT = CN()
_C.SRL.LATENT.TYPE = 'VAE'
_C.SRL.LATENT.LOSS_WEIGHT = 1
_C.SRL.LATENT.STATESIZE = 30
_C.SRL.LATENT.LEARNING_RATE = 0.001
_C.SRL.LATENT.BETA = 1.0
_C.SRL.LATENT.USE_PREPROCESSED_IMG = False
# ----------------> SSM <--------------- #
_C.SRL.SSM = CN()
_C.SRL.SSM.LOSS_WEIGHT = 1
_C.SRL.SSM.LEARNING_RATE = 0.001
_C.SRL.SSM.LATENT_1_DIM = 32
_C.SRL.SSM.LATENT_2_DIM = 256
_C.SRL.SSM.SEQ_LEN = 5
_C.SRL.SSM.LSTM_LAYERS = 2
_C.SRL.SSM.ENCODER_PRETRAINING_STEPS = 1000

_C.SRL.SSM.GAUSSIAN_NET = CN()
_C.SRL.SSM.GAUSSIAN_NET.ARCHITECTURE= [256,256]
_C.SRL.SSM.GAUSSIAN_NET.ACTIVATION= "ReLU"
_C.SRL.SSM.GAUSSIAN_NET.INIT= "orthogonal"
_C.SRL.SSM.GAUSSIAN_NET.DROPOUT= False
_C.SRL.SSM.GAUSSIAN_NET.BATCHNORM= False
_C.SRL.SSM.GAUSSIAN_NET.BIAS= True
_C.SRL.SSM.GAUSSIAN_NET.FIXED_STD= True
# ----------------> FORWARD <--------------- #
_C.SRL.FORWARD = CN()
_C.SRL.FORWARD.LOSS_WEIGHT = 1
_C.SRL.FORWARD.STOCHASTIC = False
# Parameter MLP
_C.SRL.FORWARD.ARCHITECTURE= [64,64]
_C.SRL.FORWARD.ACTIVATION= "ReLU"
_C.SRL.FORWARD.INIT= "orthogonal"
_C.SRL.FORWARD.DROPOUT= False
_C.SRL.FORWARD.BATCHNORM= False
_C.SRL.FORWARD.LEARNING_RATE = 0.001
# ----------------> INVERSE <--------------- #
_C.SRL.INVERSE = CN()
_C.SRL.INVERSE.LOSS_WEIGHT = 1
_C.SRL.INVERSE.STOCHASTIC = False
# Parameter MLP
_C.SRL.INVERSE.ARCHITECTURE= [64,64]
_C.SRL.INVERSE.ACTIVATION= "ReLU"
_C.SRL.INVERSE.INIT= "orthogonal"
_C.SRL.INVERSE.DROPOUT= False
_C.SRL.INVERSE.BATCHNORM= False
_C.SRL.INVERSE.LEARNING_RATE = 0.001
# ----------------> Reward <--------------- #
_C.SRL.REWARD = CN()
_C.SRL.REWARD.LOSS_WEIGHT = 1
_C.SRL.REWARD.STOCHASTIC = False
# Parameter MLP
_C.SRL.REWARD.ARCHITECTURE= [64,64]
_C.SRL.REWARD.ACTIVATION= "ReLU"
_C.SRL.REWARD.INIT= "orthogonal"
_C.SRL.REWARD.DROPOUT= False
_C.SRL.REWARD.BATCHNORM= False
_C.SRL.REWARD.LEARNING_RATE = 0.001
# ----------------> CURL <--------------- #
_C.SRL.CURL = CN()
_C.SRL.CURL.LOSS_WEIGHT = 1
_C.SRL.CURL.RENDER_SIZE = 100
_C.SRL.CURL.IMAGE_SIZE = 84
_C.SRL.CURL.LEARNING_RATE = 0.001
# ----------------> RECURRENT <--------------- # TODO
_C.SRL.RECURRENT = CN()
_C.SRL.RECURRENT.SEQ_LEN = 2
_C.SRL.RECURRENT.LOSS_WEIGHT = 1
_C.SRL.RECURRENT.NUM_LAYERS = 2
_C.SRL.RECURRENT.HIDDEN_SIZE = 100
# ----------------> PRIORS <--------------- #
_C.SRL.PRIORS = CN()
_C.SRL.PRIORS.LOSS_WEIGHTS = [1,1,1,1]
# ---------------------------------------------------------------- #
#                             ENCODER                              #
# ---------------------------------------------------------------- #
_C.ENCODER = CN()
_C.ENCODER.ARCHITECTURE= 'standard'
_C.ENCODER.HIDDEN_DIM= 64
_C.ENCODER.ACTIVATION= "ReLU"
_C.ENCODER.CONV_INIT= "delta_othogonal"
_C.ENCODER.LINEAR_INIT= "orthogonal"
_C.ENCODER.DROPOUT= False
_C.ENCODER.BATCHNORM= False
_C.ENCODER.BIAS= True
_C.ENCODER.NORMALIZED_LATENT = True
_C.ENCODER.SQUASHED_LATENT = True
# ---------------------------------------------------------------- #
#          REINFORCEMENT LEARNING AGENT CONFIGURATIONS             #
# ---------------------------------------------------------------- #
_C.RL = CN()
_C.RL.NAME = ""
_C.RL.FEATURE_DIM = 50
_C.RL.ADVANTAGE_NORM = False
# ----------------> CRITIC <--------------- #
_C.RL.CRITIC = CN()
# Parameter MLP
_C.RL.CRITIC.ARCHITECTURE= [64,64]
_C.RL.CRITIC.ACTIVATION= "ReLU"
_C.RL.CRITIC.INIT= "naive"
_C.RL.CRITIC.DROPOUT= False
_C.RL.CRITIC.BATCHNORM= False
# Parameter Optimizer
_C.RL.CRITIC.LEARNING_RATE= 0.01
_C.RL.CRITIC.WEIGHT_DECAY= 0
_C.RL.CRITIC.EPSILON= 0.99
_C.RL.CRITIC.UPDATE_FREQ= 1
# Parameter General
_C.RL.CRITIC.MAX_GRAD_NORM= 0
# ----------------> CRITIC <--------------- #
_C.RL.ACTOR = CN()
# Parameter MLP
_C.RL.ACTOR.ARCHITECTURE= [64,64]
_C.RL.ACTOR.ACTIVATION= "ReLU"
_C.RL.ACTOR.INIT= "naive"
_C.RL.ACTOR.DROPOUT= False
_C.RL.ACTOR.BATCHNORM= False
# Parameter Optimizer
_C.RL.ACTOR.LEARNING_RATE= 0.01
_C.RL.ACTOR.WEIGHT_DECAY= 0
_C.RL.ACTOR.EPSILON= 0.99
_C.RL.ACTOR.DETACH_FOR_CNN_UPDATE=False
_C.RL.ACTOR.UPDATE_FREQ= 2
# Parameter General
_C.RL.ACTOR.MAX_GRAD_NORM= 0
_C.RL.ACTOR.USE_TO_UPDATE = False
# ----------------> SAC <--------------- #
_C.RL.SAC = CN()
_C.RL.SAC.POLYAK = 0.995
_C.RL.SAC.GAMMA = 0.995
_C.RL.SAC.TARGET_UPDATE_FREQ = 2
# ----------------> PPO <--------------- #
_C.RL.PPO = CN()
_C.RL.PPO.FIXED_STD = False
_C.RL.PPO.MAX_KL_DIV = 0.01
_C.RL.PPO.EARLY_STOPPING = True
_C.RL.PPO.POLICY_EPOCHS = 80
_C.RL.PPO.VALUE_EPOCHS = 80
_C.RL.PPO.SQUASHED = True
_C.RL.PPO.CLIPPED_RATIO = 0.2
_C.RL.PPO.MINI_BATCH_SIZE = 125
# ----------------> PPO2 <--------------- #
_C.RL.PPO.ENTROPY_COEFFICIENT = 0
_C.RL.PPO.VALUE_COEFFICIENT = 0.5
_C.RL.PPO.CLIPPED_VALUE = True
_C.RL.PPO.TRAINING_EPOCHS = 10
_C.RL.PPO.CLIP_VALUE_RANGE = 0.2
_C.RL.PPO.MAX_GRAD_NORM = 2
_C.RL.PPO.GRAD_CLIPPING = True
_C.RL.PPO.ACTION_CLIPPING = False
_C.RL.PPO.LR_SCHEDULER = False
_C.RL.PPO.CUTOFF_COEFFICIENT = 100
# ----------------> BUFFER <--------------#
_C.RL.PPO.LAMBDA = 0.95
_C.RL.PPO.GAMMA = 0.99
_C.RL.PPO.TAU = 1
# ----------------> TRPO <--------------- #
_C.RL.TRPO = CN()
_C.RL.TRPO.CG_STEPS = 10
_C.RL.TRPO.DELTA = 0.01
_C.RL.TRPO.ALPHA = 0.8
_C.RL.TRPO.NUM_BACKTRACK = 10
_C.RL.TRPO.CG_DAMPING = 0.1
_C.RL.TRPO.CLIP_RATIO = 0.2
_C.RL.TRPO.VALUE_EPOCHS = 5
_C.RL.TRPO.ENTROPY_COEF = 0.1
_C.RL.TRPO.FIXED_STD = False
_C.RL.TRPO.SQUASHED = True
_C.RL.TRPO.LR_SCHEDULER = False
# ----------------> TRAINING <--------------- #
_C.RL.BATCH_SIZE= 128
_C.RL.UPDATE_EVERY= 100
_C.RL.UPDATE_AFTER= 1000
_C.RL.DELAYED_START= 1000
_C.RL.LEARNING_RATE= 0.001


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()