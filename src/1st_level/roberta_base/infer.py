import pickle
import os

import pandas as pd
import torch
import transformers
import tqdm

import config
import models
import dataset
import utils


def run():
    df_test = pd.read_csv(config.TEST_FILE)
    df_test.loc[:, 'dataset_label'] = df_test.text.values

    #temporary fix
    word_len =df_test.text.apply(lambda x:len(x.split()))
    df_test = df_test[word_len <= 510]
    tokenizer = config.TOKENIZER
    word_len_tokenized = df_test.text.apply(lambda x:len(tokenizer.encode(' '+' '.join(x.split())).ids)).to_numpy()
    df_test = df_test[word_len_tokenized <= 510]

    device = torch.device('cuda')
    model_config = transformers.RobertaConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    fold_models = []
    for i in range(config.N_FOLDS):
        model = models.ColeridgeModel(conf=model_config)
        model.to(device)
        model.load_state_dict(torch.load(
            f'{config.TRAINED_MODEL_PATH}/model_{i}.bin'),
            strict=False)
        model.eval()
        fold_models.append(model)

    test_dataset = dataset.ColeridgeDataset(
        texts=df_test.text.values,
        dataset_labels=df_test.dataset_label.values)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4)

    char_pred_test_start = []
    char_pred_test_end = []
    predicted_texts = []

    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            orig_text = d['orig_text']
            offsets_ = d['offsets']
            orig_start = d['orig_start']
            orig_end = d['orig_end']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            outputs_start_folds = []
            outputs_end_folds = []
            for i in range(config.N_FOLDS):
                outputs_start, outputs_end = \
                    fold_models[i](ids=ids,
                                   mask=mask,
                                   token_type_ids=token_type_ids)
                outputs_start_folds.append(outputs_start)
                outputs_end_folds.append(outputs_end)

            outputs_start = sum(outputs_start_folds) / config.N_FOLDS
            outputs_end = sum(outputs_end_folds) / config.N_FOLDS

            outputs_start = outputs_start.cpu().detach().numpy()
            outputs_end = outputs_end.cpu().detach().numpy()

            for px, original_text in enumerate(orig_text):
                offsets=offsets_[px]
            
                start_idx, end_idx = utils.get_best_start_end_idx(
                            start_logits=outputs_start[px, :], end_logits=outputs_end[px, :], 
                            orig_start=orig_start[px], orig_end=orig_end[px],)

                filtered_output = ''
                for ix in range(start_idx, end_idx + 1):
                    filtered_output += original_text[offsets[ix][0]:offsets[ix][1]]
                    if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                        filtered_output += ' '

                # Return orig tweet if it has less then 2 words
                if len(original_text.split()) < 2:
                    filtered_output = original_text

                if len(filtered_output.split()) == 1:
                    filtered_output = filtered_output.replace('!!!!', '!')
                    filtered_output = filtered_output.replace('..', '.')
                    filtered_output = filtered_output.replace('...', '.')

                filtered_output = filtered_output.replace('ïï', 'ï')
                filtered_output = filtered_output.replace('¿¿', '¿')
                
                predicted_texts.append(filtered_output.strip())

            # outputs_start = torch.softmax(outputs_start, dim=-1).cpu().detach().numpy()
            # outputs_end = torch.softmax(outputs_end, dim=-1).cpu().detach().numpy()

            # for px, text in enumerate(orig_text):
            #     char_pred_test_start.append(
            #         utils.token_level_to_char_level(text, offsets[px], outputs_start[px]))
            #     char_pred_test_end.append(
            #         utils.token_level_to_char_level(text, offsets[px], outputs_end[px]))


    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')

    with open(f'{config.INFERED_PICKLE_PATH}/roberta-predicted_texts.pkl', 'wb') as handle:
        pickle.dump(predicted_texts, handle)
    # with open(f'{config.INFERED_PICKLE_PATH}/roberta-char_pred_test_start.pkl', 'wb') as handle:
    #     pickle.dump(char_pred_test_start, handle)
    # with open(f'{config.INFERED_PICKLE_PATH}/roberta-char_pred_test_end.pkl', 'wb') as handle:
    #     pickle.dump(char_pred_test_end, handle)


if __name__ == '__main__':
    run()