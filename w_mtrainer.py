import torch
import numpy as np


def fit(train_loader, val_loader,embedding_net, model, loss_fn, loss2_fn, optimizer, scheduler, n_epochs, cuda, log_interval,metrics=[],start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, loss2_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        # for metric in metrics1:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, val_loss1, val_loss2, metrics = test_epoch(val_loader, model, loss_fn, loss2_fn, cuda, metrics)
        val_loss /= len(val_loader)
        val_loss1 /= len(val_loader)
        val_loss2 /= len(val_loader)
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}.loss1:{:.4f}.loss2:{:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss,val_loss1,val_loss2)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        # for metric in metrics1:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())
        print(message)
        if epoch % 11== 0:
            #torch.save(embedding_net.state_dict(), 'shiyan.pkl')
            torch.save(model.state_dict(), 'vgg_gram_fi8.pkl')

def train_epoch(train_loader, model, loss_fn,loss2_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()
    # for metric in metrics1:
    #     metric.reset()

    model.train()
    losses = []
    losses1 = []
    losses2 = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()

        outputs, cls = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        if type(cls) not in (tuple, list):
            cls = (cls,)

        loss_inputs = cls

        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(outputs[0], cls[0], target[0])
        loss2_outputs = loss2_fn(cls[0], target[0])

        loss1 = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        loss2 = loss2_outputs[0] if type(loss2_outputs) in (tuple, list) else loss2_outputs
        loss = 0.2 * loss1 + 0.8 * loss2
        losses.append(loss.item())
        losses1.append(loss1.item())
        losses2.append(loss2.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(cls, target, loss2_outputs)
        # for metric in metrics1:
        #     metric(cls, target, loss2_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses),np.mean(losses1),np.mean(losses2))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())
            # for metric in metrics1:

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, loss2_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        # for metric in metrics1:
        #     metric.reset()
        model.eval()
        val_loss = 0
        val_loss1 = 0
        val_loss2 = 0

        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs, cls = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            if type(cls) not in (tuple, list):
                cls = (cls,)
            loss_inputs = outputs

            if target is not None:
                target = (target,)
                loss_inputs += target
            loss_outputs = loss_fn(outputs[0], cls[0], target[0])
            #loss2_outputs = loss2_fn(outputs[0], target[0].long().cpu())
            loss2_outputs = loss2_fn(cls[0], target[0])
            loss1 = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            loss2 = loss2_outputs[0] if type(loss2_outputs) in (tuple, list) else loss2_outputs
            loss = 0.2 * loss1 + 0.8 * loss2

            val_loss += loss.item()
            val_loss1 += loss1.item()
            val_loss2 += loss2.item()
            for metric in metrics:
                metric(cls, target, loss2_outputs)
            # for metric in metrics1:
            #     metric(cls, target, loss2_outputs)
    return val_loss, val_loss1, val_loss2, metrics
