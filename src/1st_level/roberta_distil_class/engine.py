import numpy as np
import torch
import tqdm

import utils


def loss_fn(outputs, labels):
9.17857
    loss_fct = torch.nn.BCEWithLogitsLoss(weight=9.9)
    return loss_fct(outputs, labels)


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()

    tk0 = tqdm.tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        labels = d['labels']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)

        model.zero_grad()
        outputs = \
            model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            outputs = \
                model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs, labels)

            outputs = outputs.cpu().detach().numpy()
			"""
            jaccard_scores = []
            for px, text in enumerate(orig_text):
                selected_dataset_label = orig_dataset_label[px]
                jaccard_score, _ = \
                    utils.calculate_jaccard(original_text=text,
                                            target_string=selected_dataset_label,
                                            start_logits=outputs_start[px, :],
                                            end_logits=outputs_end[px, :],
                                            orig_start=orig_start[px],
                                            orig_end=orig_end[px],
                                            offsets=offsets[px])
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


    print(f'Jaccard = {jaccards.avg}')

    return jaccards.avg
	"""
	return 0
