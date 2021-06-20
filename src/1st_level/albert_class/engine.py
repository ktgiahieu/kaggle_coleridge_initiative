import numpy as np
import torch
import tqdm

import utils


def loss_fn(outputs, labels, weight):
    loss_fct = torch.nn.BCEWithLogitsLoss(weight=weight)
    return loss_fct(outputs, labels)


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()

    tk0 = tqdm.tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        mask = d['mask']
        labels = d['labels']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)

        model.zero_grad()
        outputs = \
            model(ids=ids, mask=mask)
        
        weight = torch.Tensor([10]).to(device, dtype=torch.float)
        loss = loss_fn(outputs, labels ,weight)
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
            mask = d['mask']
            labels = d['labels']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            outputs = \
                model(ids=ids, mask=mask)
            loss = loss_fn(outputs, labels)

            outputs = outputs.cpu().detach().numpy()
    return 0
