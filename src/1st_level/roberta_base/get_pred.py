import pickle

with open('/content/kaggle_tweet/pickles/roberta-char_pred_test_start.pkl', 'wb') as handle:
    char_pred_test_start = pickle.load(handle)
with open('/content/kaggle_tweet/pickles/roberta-char_pred_test_end.pkl', 'wb') as handle:
    char_pred_test_end = pickle.dump(handle)
print(len(char_pred_test_start), len(char_pred_test_end))

