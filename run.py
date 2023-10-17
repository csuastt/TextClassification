import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import sys

# 定义模型
if len(sys.argv) != 2:
    print("The number of arguments should be 2.")
    raise BaseException

if sys.argv[1] == "cnn":
    from model.model import CNN, CNNConfig
    # 参数
    config = CNNConfig()
    # 设定随机种子
    torch.manual_seed(config.seed)            # 为CPU设置随机种子
    np.random.seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)   # 为所有GPU设置随机种子
    # 读取权重
    weights = torch.load("./dataset/weights.pt")
    # 例化模型
    model = CNN(config, weights)
    # 载入参数
    model.load_state_dict(torch.load("./model/cnn.pt"))
    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay
    )
elif sys.argv[1] == "cnn_att":
    from model.model import CNN, CNNConfig
    # 参数
    config = CNNConfig()
    config.is_att = True
    # 设定随机种子
    torch.manual_seed(config.seed)            # 为CPU设置随机种子
    np.random.seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)   # 为所有GPU设置随机种子
    # 读取权重
    weights = torch.load("./dataset/weights.pt")
    # 例化模型
    model = CNN(config, weights)
    # 载入参数
    model.load_state_dict(torch.load("./model/cnn_att.pt"))
    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay
    )
elif sys.argv[1] == "cnn_inception":
    from model.model import CNNInception, CNNConfig
    # 参数
    config = CNNConfig()
    # 设定随机种子
    torch.manual_seed(config.seed)            # 为CPU设置随机种子
    np.random.seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)   # 为所有GPU设置随机种子
    # 读取权重
    weights = torch.load("./dataset/weights.pt")
    # 例化模型
    model = CNNInception(config, weights)
    # 载入参数
    model.load_state_dict(torch.load("./model/cnn_inception.pt"))
    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay
    )
elif sys.argv[1] == "rnn":
    from model.model import RNN, RNNConfig
    # 参数
    config = RNNConfig()
    # 设定随机种子
    torch.manual_seed(config.seed)            # 为CPU设置随机种子
    np.random.seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)   # 为所有GPU设置随机种子
    # 读取权重
    weights = torch.load("./dataset/weights.pt")
    # 例化模型
    model = RNN(config, weights)
    # 载入参数
    model.load_state_dict(torch.load("./model/rnn.pt"))
    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay
    )
elif sys.argv[1] == "baseline":
    from model.model import FastText, FastTextConfig
    # 参数
    config = FastTextConfig()
    # 设定随机种子
    torch.manual_seed(config.seed)            # 为CPU设置随机种子
    np.random.seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)   # 为所有GPU设置随机种子
    # 读取权重
    weights = torch.load("./dataset/weights.pt")
    # 例化模型
    model = FastText(config, weights)
    # 载入参数
    model.load_state_dict(torch.load("./model/baseline.pt"))
    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay
    )
else:
    print("Model not found!")
    raise BaseException


# 载入测试集
# 读入数据
x_train = torch.load("./dataset/train_data_x.pt")
y_train = torch.load("./dataset/train_data_y.pt")
y_train = torch.topk(y_train, 1)[1].squeeze(1)
x_test = torch.load("./dataset/test_data_x.pt")
y_test = torch.load("./dataset/test_data_y.pt")
y_test = torch.topk(y_test, 1)[1].squeeze(1)
x_vali = torch.load("./dataset/vali_data_x.pt")
y_vali = torch.load("./dataset/vali_data_y.pt")
y_vali = torch.topk(y_vali, 1)[1].squeeze(1)
# 数据集
train_data = TensorDataset(x_train, y_train) 
test_data = TensorDataset(x_test, y_test) 
vali_data = TensorDataset(x_vali, y_vali)
train_dataloader = DataLoader(
    train_data, 
    batch_size=config.batch_size,
    shuffle=True
)
test_dataloader = DataLoader(
    test_data,
    batch_size=config.batch_size
)
vali_dataloader = DataLoader(
    vali_data,
    batch_size=config.batch_size
)

def train_loop(dataloader, model, loss_fn, optimizer, config, output=True):
    '''
    训练循环
    '''
    size = len(dataloader.dataset)
    cnt = 0
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # GPU训练
        if config.cuda:
            X = X.to("cuda")
            y = y.to("cuda")
        # 预测和计算损失
        pred = model(X)
        loss = loss_fn(pred, y)
        # 记录正确数量
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        cnt += y.size(0)
        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 70 == 0:
        if batch * len(X) == 4480:
            # loss, current = loss.item(), batch * len(X)
            current = batch * len(X)
            # correct = (pred.argmax(1) == y).type(torch.float).sum().item()/\
            #     y.size(0)
            if output:
                print(f"Acc: {(100*correct/cnt):>0.1f}% \
                    Avg loss: {train_loss/cnt:>7f}  [{current:>5d}/{size:>5d}]")
    return train_loss/cnt, correct/cnt

def test_loop(dataloader, model, loss_fn, is_test, config, output=True):
    '''
    测试或验证循环
    '''
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # GPU训练
            if config.cuda:
                X = X.to("cuda")
                y = y.to("cuda")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    if is_test and output:
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, \
            Avg loss: {test_loss:>8f} \n"
        )
    elif output:
        print(
            f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, \
            Avg loss: {test_loss:>8f} \n"
        )
    return test_loss, correct

# 直接测试
# 计算三项指标和混淆矩阵
from sklearn.metrics import classification_report, \
    precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

# 统计
if config.cuda:
    model.to("cuda")
y_true = []
y_pred = []
X_test = []
with torch.no_grad():
    for X, y in test_dataloader:
        # GPU训练
        if config.cuda:
            X = X.to("cuda")
            y = y.to("cuda")
        pred = model(X)
        y_pred.extend(pred.argmax(1).cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
        X_test.append(X)


print(classification_report(
    y_true, y_pred, 
    labels=[0, 1, 2, 3, 4, 5, 6],
    target_names=['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame'],
    digits=3
))
precision_recall_fscore_support(y_true, y_pred, average='micro')
# 绘制混淆矩阵
cm = confusion_matrix(y_true, y_pred, normalize='true',labels=[0,1,2,3,4,5,6])
df = pd.DataFrame(cm, columns=['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame'],index=['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame'])
sns.set()
sns.heatmap(df,annot=True)
