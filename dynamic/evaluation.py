import time
import torch
import os
import sys
from tools import AverageMeter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score


def accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct = correct.float().sum().item()

    return n_correct / batch_size


def train_epoch(epoch, data_loader, model, device, criterion, optimizer, logger):
    print(f'Training epoch %d' % (epoch))

    model.train()

    time_per_batch = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    print_per_n_batch = 10 
    
    start_time = time.time()
    batch_start_time = time.time()

    preds = []
    truth = []
    ds_size = len(data_loader)

    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)

        _, pred = outputs.topk(1, 1)

        preds.extend(pred.squeeze().cpu().numpy())
        truth.extend(targets.cpu().numpy())

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_start_time
        time_per_batch.update(batch_time)
        batch_start_time = time.time()

        if i % print_per_n_batch == 0:
            print(f'Epoch {epoch}: [{i}/{ds_size}] \tTime: {batch_time:.3f} ({time_per_batch.avg:.3f})\
                  \tLoss: {losses.val:.3f} ({losses.avg:.3f})\tAcc: {acc:.3f} ({accs.avg:.3f})')
    
            sys.stdout.flush()
    
    prec = precision_score(truth, preds, average='macro')
    recall = recall_score(truth, preds, average='macro')
    bacc = balanced_accuracy_score(truth, preds)
    total_time = time.time() - start_time

    logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accs.avg,
        'precision': prec,
        'recall': recall,
        'bacc': bacc
    })

    print(f'Epoch: [{epoch}]\tTime: {total_time:.3f}\tLoss: {losses.avg:.3f}\tAcc:\
          {accs.avg:.3f} \tPrecision: {prec:.3f}\tRecall: {recall:.3f}\tBACC: {bacc:.3f}')

    sys.stdout.flush()

def validate_epoch(epoch, data_loader, model, device, criterion, logger):
    print(f'Validating epoch %d' % (epoch))

    model.eval()

    time_per_batch = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    print_per_n_batch = 10 
    
    start_time = time.time()
    batch_start_time = time.time()

    preds = []
    truth = []
    ds_size = len(data_loader)

    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)

        _, pred = outputs.topk(1, 1)

        preds.extend(pred.squeeze().cpu().numpy())
        truth.extend(targets.cpu().numpy())

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))

        batch_time = time.time() - batch_start_time
        time_per_batch.update(batch_time)
        batch_start_time = time.time()

        if i % print_per_n_batch == 0:
            print(f'Epoch {epoch}: [{i}/{ds_size}] \tTime: {batch_time:.3f} ({time_per_batch.avg:.3f})\
                  \tLoss: {losses.val:.3f} ({losses.avg:.3f})\tAcc: {acc:.3f} ({accs.avg:.3f})')
            
            sys.stdout.flush()

    prec = precision_score(truth, preds, average='macro')
    recall = recall_score(truth, preds, average='macro')
    bacc = balanced_accuracy_score(truth, preds)
    total_time = time.time() - start_time

    logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accs.avg,
        'precision': prec,
        'recall': recall,
        'bacc': bacc
    })

    print(f'Val Epoch: [{epoch}]\tTime: {total_time:.3f}\tLoss: {losses.avg:.3f}\tAcc:\
          {accs.avg:.3f} \tPrecision: {prec:.3f}\tRecall: {recall:.3f}\tBACC: {bacc:.3f}')
    sys.stdout.flush()

    return losses.avg, accs.avg, prec, recall, bacc


def test_model(data_loader, model, device, criterion, logger):
    print('Testing model')

    model.eval()

    time_per_batch = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    print_per_n_batch = 10 
    
    start_time = time.time()
    batch_start_time = time.time()

    preds = []
    truth = []
    ds_size = len(data_loader)

    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)

        _, pred = outputs.topk(1, 1)

        preds.extend(pred.squeeze().cpu().numpy())
        truth.extend(targets.cpu().numpy())

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))

        batch_time = time.time() - batch_start_time
        time_per_batch.update(batch_time)
        batch_start_time = time.time()

        if i % print_per_n_batch == 0:
            print(f'Batch: [{i}/{ds_size}] \tTime: {batch_time:.3f} ({time_per_batch.avg:.3f})\
                  \tLoss: {losses.val:.3f} ({losses.avg:.3f})\tAcc: {acc:.3f} ({accs.avg:.3f})')
            
            sys.stdout.flush()

    prec = precision_score(truth, preds, average='macro')
    recall = recall_score(truth, preds, average='macro')
    bacc = balanced_accuracy_score(truth, preds)
    total_time = time.time() - start_time

    logger.log({
        'loss': losses.avg,
        'acc': accs.avg,
        'precision': prec,
        'recall': recall,
        'bacc': bacc
    })

    print(f'Testing:\tTime: {total_time:.3f}\tLoss: {losses.avg:.3f}\tAcc:\
          {accs.avg:.3f} \tPrecision: {prec:.3f}\tRecall: {recall:.3f}\tBACC: {bacc:.3f}')
    sys.stdout.flush()

    return losses.avg, accs.avg, prec, recall, bacc