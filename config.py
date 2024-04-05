from pathlib import Path

DATA_PATH = Path('./data')
TRAIN_PATH = DATA_PATH / 'train'
TEST_PATH = DATA_PATH / 'test'

NUM_WORKERS = 0
DEVICE = 'cuda'
Z_DIM = 100
BATCH_SIZE = 128
LR = 0.0002

BETA_1 = 0.5
BETA_2 = 0.999

DISPLAY_STEP = 500

EPOCHS = 20