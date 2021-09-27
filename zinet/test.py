
import os
import numpy as np

from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from tqdm import trange

from zi_net import ZiNet

# http://yann.lecun.com/exdb/mnist/
# 加载文件
def fectch(url):
    import requests,gzip,os,hashlib,numpy
    
    fp = os.path.join("/tmp",hashlib.md5(url.encode('utf-8')).hexdigest())
    # print(fp)

    if os.path.isfile(fp):
        print("exist")
        with open(fp,"rb") as f:
            dat = f.read()
    else:
        print("hello")
        with open(fp,"wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat),dtype=np.uint8).copy()

# def fectch_labels(file_path):
    

if __name__ == "__main__":

    # url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"

    # fp = os.path.join("/tmp",hashlib.md5(url.encode('utf-8')).hexdigest())
    # print(os.path.isfile(fp))
    # download data
    X_train = fectch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1,28,28))
    # print(X_train.shape)
    # Y_train = fectch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    # print(Y_train.shape)
    # X_test = fectch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1,28,28))
    # Y_test = fectch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    print(X_train.shape)

    """

    model = ZiNet()

    # pred = model(torch.tensor(X_train[0:10].reshape((-1,28*28))).float())
    # print(pred)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    batch_size = 32

    t = trange(100)

    print("hello...")
    print(X_train.shape)

    for i in  t:
        samp = np.random.randint(0,X_train.shape[0],size=(batch_size))
        
        X = torch.tensor(X_train[samp]).reshape((-1,28*28)).float()
        Y = torch.tensor(Y_train[samp]).long()
        optim.zero_grad()
        pred = model(X)
        cat = torch.argmax(pred,dim=1)
        accuracy = (cat == Y).float().mean()
        loss =  loss_fn(pred,Y)
        loss.backward()
        optim.step()

        t.set_description("loss %.2f accuracy %.2f" % (loss.item(),accuracy.item()))
        t.refresh()

    """