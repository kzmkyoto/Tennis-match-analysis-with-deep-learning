import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from TrackNet_utils import postprocess
from scipy.spatial import distance


def train(model, train_loader, loss_fn, optimizer, device, epoch, max_iters):
    # loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    losses = []
    # Set the model to training mode
    model.train()

    for iter_id, batch in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(batch[0].to(device))
        ground_truth = batch[1].to(device)

        loss = loss_fn(outputs, ground_truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        end_time = time.time()
        duration = end_time - start_time

        losses.append(loss.item())

        print(f'epoch:{epoch}, iteration:{iter_id}/{max_iters}, loss:{round(loss, 5)}, time:{duration}')

        if iter_id >= max_iters - 1:
            break
    
    return np.mean(losses)


# Test the accuracy, precision, recall and f1-measure of the prediction
def validate(model, val_loader, device, epoch, min_dist=5):
    '''
    tp: True Positive
    fp: False Positive
    tn: True Negative
    fn: False Negative
    '''
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    
    criterion = nn.CrossEntropyLoss()
    model.eval()

    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            out = model(batch[0].float().to(device))
            gt = torch.tensor(batch[1], dtype=torch.long, device=device)
            loss = criterion(out, gt)
            losses.append(loss.item())

            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                x_pred, y_pred = postprocess(output[i])
                x_gt = batch[2][i]
                y_gt = batch[3][i]
                vis = batch[4][i]

                if x_pred:
                    if vis != 0:
                        dist = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dist < min_dist:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:
                        fp[vis] += 1
                
                if not x_pred:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1
    
    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    print(f"precision : {precision}")
    print(f"recall : {recall}")
    print(f"f1 : {f1}")

    return np.mean(losses), precision, recall, f1

