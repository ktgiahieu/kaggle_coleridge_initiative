import tokenizers
import os
is_kaggle = 'KAGGLE_URL_BASE' in os.environ

# Paths
if is_kaggle:
    comp_name = 'coleridgeinitiative-show-us-the-data'
    my_restructured_dataset = 'my-coleridgeinitiative-show-us-the-data'
    my_impl_dataset = 'my-coleridge-initiative-impl'
    my_model_dataset = 'my-roberta-base-squad2-qa-model'

    TOKENIZER_PATH = f'../input/{my_impl_dataset}/src/1st_level/roberta_tokenizer'
    TRAINING_FILE = f'../input/{my_restructured_dataset}/data/train_folds.csv'
    TEST_FILE = 'test.csv'
    SUB_FILE = f'../input/{comp_name}/sample_submission.csv'
    MODEL_SAVE_PATH = f'../input/{my_model_dataset}'
    TRAINED_MODEL_PATH = f'../input/{my_model_dataset}'
    INFERED_PICKLE_PATH = '.'

    MODEL_CONFIG = '../input/my-roberta-base-squad2'
else: #colab
    repo_name = 'kaggle_coleridge_initiative'
    drive_name = 'ColeridgeInitiative'
    
    TOKENIZER_PATH = f'/content/{repo_name}/src/1st_level/roberta_tokenizer'
    TRAINING_FILE = f'/content/{repo_name}/data/train_folds.csv'
    TEST_FILE = f'/content/{repo_name}/data/test.csv'
    SUB_FILE = f'/content/{repo_name}/data/sample_submission.csv'
    MODEL_SAVE_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save'
    TRAINED_MODEL_PATH = f'/content/gdrive/MyDrive/Dataset/{drive_name}/model_save'
    INFERED_PICKLE_PATH = f'/content/{repo_name}/pickle'

    MODEL_CONFIG = 'deepset/roberta-base-squad2'

# Model params
SEED = 25
N_FOLDS = 5
EPOCHS = 1
LEARNING_RATE = 4e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
MAX_LEN = 512  # actually = inf
print(f'{TOKENIZER_PATH}/vocab.json')
TOKENIZER = tokenizers.ByteLevelBPETokenizer.from_file(
    vocab_filename=f'{TOKENIZER_PATH}/vocab.json',
    merges_filename=f'{TOKENIZER_PATH}/merges.txt',
    lowercase=True,
    add_prefix_space=True)
HIDDEN_SIZE = 768
N_LAST_HIDDEN = 12
HIGH_DROPOUT = 0.5
SOFT_ALPHA = 0.6
WARMUP_RATIO = 0.25
WEIGHT_DECAY = 0.001
USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30
