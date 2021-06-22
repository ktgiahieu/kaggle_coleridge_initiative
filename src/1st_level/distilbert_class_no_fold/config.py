import tokenizers
from transformers import DistilBertTokenizerFast
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
if is_kaggle:
    comp_name = 'coleridgeinitiative-show-us-the-data'
    my_restructured_dataset = 'coleridge-classify-train-folds'
    my_impl_dataset = 'my-coleridge-initiative-impl'
    my_model_dataset = 'my-distilbert-class-noise-model'

    TOKENIZER_PATH = f'../input/{my_impl_dataset}/src/1st_level/roberta_tokenizer'
    TRAINING_FILE = f'../input/{my_restructured_dataset}/train_folds.csv'
    TEST_FILE = 'test_all_data.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'./'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/my-distilbert-base-cased-fast'
else: #colab
    repo_name = 'kaggle_coleridge_initiative'
    drive_name = 'ColeridgeInitiative'
    
    #TOKENIZER_PATH = f'/content/{repo_name}/src/1st_level/roberta_tokenizer'
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds.csv'
    TEST_FILE = f'test_all_data.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/distilbert_class'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save/1st_level/distilbert_class'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    #MODEL_CONFIG = 'huawei-noah/TinyBERT_General_4L_312D'
    MODEL_CONFIG = 'distilbert-base-cased'

# Model params
SEED = 25
N_FOLDS = 5
EPOCHS = 4
LEARNING_RATE = 5e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
MAX_LEN = 64  # actually = 434
TOKENIZER = DistilBertTokenizerFast.from_pretrained(
    MODEL_CONFIG)
HIDDEN_SIZE = 768
HIGH_DROPOUT = 0.5
WARMUP_RATIO = 0.25
WEIGHT_DECAY = 0.001
USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30
