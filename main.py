import scipy.io as io

import torch
import torchvision
import os

from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from mtrainer import *
import numpy as np
from datahd import *
cuda = torch.cuda.is_available()
from datasets import BalancedBatchSampler
import matplotlib
import matplotlib.pyplot as plt
#from torchvision.datasets import MNIST
from torchvision import transforms
from networks import *
mean, std = 0.1307, 0.3081
import torch.nn as nn
#from resnet import *
# from inception_v3 import *
from resnet import *
from googlenet import *
#from old_losses import TripletLoss, Accuracy, OnlineTripletLoss
from w_losses import *
#from oldutils import *
from w_utils import *
from w_mtrainer import *
from datasets import BalancedBatchSampler
from metrics import AccumulatedAccuracyMetric, AverageNonzeroTripletsMetric
from googlenet import *


# cmats=[]
# accuracy=[]


def extract_embeddings(dataloader, model,name):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 544))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.forward(images)[0].data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
        # f = open("newours_0.1_0.9_train.txt", 'a')
        # np.savetxt("newours_0.1_0.9_train.txt", embeddings)
        io.savemat(name,{'data':embeddings})
    return embeddings, labels


def extract_embeddingstest(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zros((len(dataloader.dataset), 2048))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.forward(images)[0].data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k+=len(images)
        f = open("newours_0.1_0.9_test.txt", 'a')
        np.savetxt("newours_0.1_0.9_test.txt", embeddings)
    return embeddings, labels


embedding_net = Inception3()
# embedding_net = ResNet()
# embedding_net = inception_v3()

# embedding_net.load_state_dict(torch.load('res_e6.pkl'))
#

model = embedding_net
# torch.save(model.state_dict(), 'res_7.pkl')


# for i in range(1, 6):
#     imodel=model
    # torch.save(imodel.state_dict(), 'res_7.pkl')
    # print(model)
    # model.load_state_dict(torch.load('res_7.pkl'))
train_dataset = MyDataset(root='/home/ubuntu5/yxx/Benchmark_split/Fi/', datatxt='fi_trainimg_label.txt', train=True, transform=transforms.Compose([
                           transforms.Scale(356),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(299),
                           transforms.ToTensor(),
                           #transforms.Normalize((mean,), (std,))
                       ]), target_transform=transforms.Compose([transforms.ToTensor()]))

test_dataset = MyDataset(root='/home/ubuntu5/yxx/Benchmark_split/Fi/', datatxt='fi_testimg_label.txt', transform=transforms.Compose([
                           # transforms.Resize((28, 28)),
                           transforms.Scale(356),
                           transforms.CenterCrop(299),
                           transforms.ToTensor(),
                           #transforms.Normalize((mean,), (std,))
                       ]), target_transform=transforms.Compose([transforms.ToTensor()]))

train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=8, n_samples=6)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=8, n_samples=6)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

kwargs = {'num_workers': 6, 'pin_memory': True} if cuda else {}

online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)

online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)


# from networks import EmbeddingNet, TripletNet, ResNet



if cuda:
    model.cuda()

margin1 = 0.2
margin2 = 0.1

loss_fn1 = OnlineTripletLoss(margin1, margin2, SemihardNegativeTripletSelector(margin1, margin2))

loss_fn2 = nn.CrossEntropyLoss()

lr = 1e-2

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = lr_scheduler.StepLR(optimizer, 40, gamma=0.1, last_epoch=-1)
n_epochs = 100
log_interval = 100

# predict_label1 = filt(test_loader=test_loader, model=model, scheduler=scheduler, num_epochs=1, cuda=cuda)

fit(online_train_loader, online_test_loader, embedding_net, model, loss_fn1, loss_fn2, optimizer, scheduler, n_epochs, cuda, log_interval,  metrics=[AccumulatedAccuracyMetric()])

    # a,b=extract_embeddings(train_loader,imodel,'./data_fea/res_iap_train_'+str(i)+'.mat')
    # c,d=extract_embeddings(test_loader,imodel,'./data_fea/res_iap_test'+str(i)+'.mat')
    # del model, embedding_net, loss_fn1, loss_fn2, online_train_loader, online_test_loader


#     labels = test_dataset.__labels__()
#     labels_list=labels.tolist()
#     io.savemat('./pre_label_res/abs_label_'+str(i)+'.mat', {'data':labels_list})
#     acc=fitest(test_loader, model, embedding_net, loss_fn2, scheduler, 1, cuda, metrics=[AccumulatedAccuracyMetric()])
#     predict_label1 = filt(test_loader=test_loader, model=model, scheduler=scheduler, num_epochs=1, cuda=cuda)
#     io.savemat('./pre_label_res/abs_pl_'+str(i)+'.mat', {'data': predict_label1})  #########
#     # predict_label1=np.transpose(np.array(predict_label1))
#     labels = np.array(labels)
#     labels = labels.astype(int)
#
#     cmat = confusion_matrix(labels, predict_label1)
#     cmats.append(cmat)
#     accuracy.append(acc)
# np.save('.pre_label_res/abs_cmats.npy', np.array(cmats))
# np.save('pre_label_res/abs_accs.npy', accuracy)
# for j in range(5):
#     cmat=cmats[j]
#     for i in range(len(cmat)):
#         print(cmat[i] / np.sum(cmat[i]))
#     for i in range(len(cmat)):
#         print((cmat[i] / np.sum(cmat[i]))[i])
# print(accuracy)


#fitest(test_loader, embedding_net, model, loss_fn2, scheduler, n_epochs, cuda, metrics=[AccumulatedAccuracyMetric()])




# train_embeddings_tl,train_labels_tl = extract_embeddings(train_loader,embedding_net)
# val_embeddings_tl,val_labels_tl = extract_embeddingstest(test_loader,embedding_net)