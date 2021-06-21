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
    df_test.loc[:, 'label'] = 0

    device = torch.device('cuda')
    model_config = transformers.DistilBertConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    model = models.ColeridgeModel(conf=model_config)
    model.to(device)
    model.load_state_dict(torch.load(
        f'{config.TRAINED_MODEL_PATH}/model_all.bin'),
        strict=False)
    model.eval()

    test_dataset = dataset.ColeridgeDataset(
        texts=df_test.text.values,
        labels=df_test.label.values)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2)
    
    predicted_labels = []

    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            outputs = \
                model(ids=ids, mask=mask)

            outputs = outputs.cpu().detach().numpy()
			      #outputs = torch.sigmoid(outputs)
            predicted_labels.extend(outputs.squeeze(-1).tolist())


    if not os.path.isdir(f'{config.INFERED_PICKLE_PATH}'):
        os.makedirs(f'{config.INFERED_PICKLE_PATH}')

    with open(f'{config.INFERED_PICKLE_PATH}/distilbert-predicted_labels.pkl', 'wb') as handle:
        pickle.dump(predicted_labels, handle)
    # with open(f'{config.INFERED_PICKLE_PATH}/roberta-char_pred_test_start.pkl', 'wb') as handle:
    #     pickle.dump(char_pred_test_start, handle)
    # with open(f'{config.INFERED_PICKLE_PATH}/roberta-char_pred_test_end.pkl', 'wb') as handle:
    #     pickle.dump(char_pred_test_end, handle)


if __name__ == '__main__':
    run()
