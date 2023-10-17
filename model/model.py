import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型定义

class CNN(nn.Module):
    def __init__(self, args, weights):
        '''
        args - 参数
        weights - 预训练权重
        is_att - 是否使用attention
        '''
        super(CNN, self).__init__()
        # 参数设定
        # 词汇表大小
        self.num_word = args.num_word
        # 词向量长度
        self.dim = args.dim
        # 类别数量
        self.num_class = args.num_class
        # 输入channel
        self.num_chan = args.num_chan
        # 卷积核的有关参数
        self.num_kerl = args.num_kerl
        self.kerl_sizes = args.kerl_sizes

        # 定义各层
        # 不使用预训练词向量
        # self.embed = nn.Embedding(self.num_word, self.dim)
        # 使用预训练词向量
        self.embed = nn.Embedding.from_pretrained(weights)
        # 训练过程中对权重进行微调
        self.embed.weight.requires_grad = True
        # 卷积层和线性层
        if args.is_att:
            self.convs = nn.ModuleList(
                [nn.Conv2d(
                    self.num_chan,
                    self.num_kerl,
                    (kerl_size, self.dim * 2)
                ) for kerl_size in self.kerl_sizes]
            )
        else:
            self.convs = nn.ModuleList(
                [nn.Conv2d(
                    self.num_chan,
                    self.num_kerl,
                    (kerl_size, self.dim)
                ) for kerl_size in self.kerl_sizes]
            )
        self.fc = nn.Linear(
            self.num_kerl * len(self.kerl_sizes), 
            self.num_class
        )
        self.dropout = nn.Dropout(args.dropout)

        # attention部分
        self.is_att = args.is_att
        if self.is_att:
            self.att_w = nn.Parameter(
                torch.zeros(self.dim)
            )
            self.tanh = nn.Tanh()

    def forward(self, x):
        # 词嵌入
        x = self.embed(x)
        # attention
        if self.is_att:
            H = x
            M = self.tanh(H)
            a = F.softmax(
                torch.matmul(M, self.att_w),
                dim=1
            ).unsqueeze(-1)
            x = torch.cat((H * a, x), 2)
        # 卷积操作
        x = x.unsqueeze(1)
        x = [
            F.relu(conv(x)).squeeze(3) 
            for conv in self.convs
        ] 
        # 池化
        x = [
            F.max_pool1d(row, row.size(2)).squeeze(2)
            for row in x
        ]  
        # 输出结果
        x = torch.cat(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class RNN(nn.Module):
    '''
    用双向LSTM实现RNN,
    加了池化层拥有CNN类似的视野能力
    '''
    def __init__(self, args, weights):
        super(RNN, self).__init__()
        # 参数设定
        # 词汇表大小
        self.num_word = args.num_word
        # 词向量长度
        self.dim = args.dim
        # 类别数量
        self.num_class = args.num_class
        # 隐藏层大小
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers

        # 定义各层
        # 不使用预训练词向量
        # self.embed = nn.Embedding(self.num_word, self.dim)
        # 使用预训练词向量
        self.embed = nn.Embedding.from_pretrained(weights)
        # 训练过程中对权重进行微调
        self.embed.weight.requires_grad = True
        # lstm
        self.lstm = nn.LSTM(
            self.dim, 
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout
        )
        self.max_pool = nn.MaxPool1d(179)
        self.fc = nn.Linear(
            self.hidden_size * 2 + self.dim,
            self.num_class
        )

    def forward(self, x):
        # 词嵌入
        tmp = self.embed(x)
        # lstm
        x, _ = self.lstm(tmp)
        # 池化
        x = torch.cat(
            (tmp, x), 2
        )
        x = F.relu(x)
        # 输出结果
        x = x.permute(0, 2, 1)
        x = self.max_pool(x).squeeze()
        x = self.fc(x)
        if x.ndim < 2:
            x = x.unsqueeze(0)
        x = F.log_softmax(x, dim=1)
        return x


class FastText(nn.Module):
    '''
    简单的FastText作为Baseline
    '''
    def __init__(self, args, weights):
        super(FastText, self).__init__()
        # 参数设定
        # 词汇表大小
        self.num_word = args.num_word
        # 词向量长度
        self.dim = args.dim
        # 类别数量
        self.num_class = args.num_class
        # 隐藏层大小
        self.hidden_size = args.hidden_size
        # ngram词表大小
        self.num_gram = args.num_gram

        # 定义各层
        # 不使用预训练词向量
        # self.embed = nn.Embedding(self.num_word, self.dim)
        # 使用预训练词向量
        self.embed = nn.Embedding.from_pretrained(weights)
        # 训练过程中对权重进行微调
        self.embed.weight.requires_grad = True
        # 其他的gram
        self.embed_2gram = nn.Embedding(self.num_gram, self.dim)
        self.embed_3gram = nn.Embedding(self.num_gram, self.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(3 * self.dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_class)

        # test
        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.num_class)
        )
    
    def ngramHash(self, front, back):
        '''
        N-Gram哈希函数
        '''
        return (front * 19260817 + back) % self.num_gram

    def forward(self, x):
        # ngram hash
        # y = torch.roll(x, 1, 1)
        # z = torch.roll(y, 1, 1)
        # # # 词嵌入
        # one_gram = self.embed(x)
        # y = y.map_(x, self.ngramHash)
        # two_gram = self.embed_2gram(y)
        # z = z.map_(y, self.ngramHash)
        # tri_gram = self.embed_3gram(z) # [batch_size, 179, 300]
        # 合并
        # x = torch.cat(
        #     (one_gram, two_gram, tri_gram),
        #     2
        # )
        # # 输出结果
        # x = x.mean(dim=1)
        # x = self.dropout(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)

        # test
        x = self.embed(x)
        x = self.fc(torch.mean(x, dim=1))
        x = F.log_softmax(x, dim=1)

        return x


class InceptionLayer(nn.Module):
    '''
    感知层
    '''
    def __init__(self, in_chan, out_chan):
        super(InceptionLayer, self).__init__()
        # 几个分支
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_chan, out_chan//4, 1),
        )
        self.branch2 = nn.Sequential(
            nn.MaxPool1d(3, 1, 1),
            nn.Conv1d(in_chan, out_chan//4, 1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_chan, out_chan//4, 1),
            nn.Conv1d(
                out_chan//4, 
                out_chan//4, 
                3, 1, 1
            )
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_chan, out_chan//4, 3, 1, 1),
            nn.Conv1d(
                out_chan//4, 
                out_chan//4, 
                5, 1, 2
            )
        )
        self.out = nn.Sequential(
            nn.ReLU(True)
        )

    def forward(self, x):
        x = torch.cat(
            (
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
            ), 1
        )
        x = self.out(x)
        return x


class CNNInception(nn.Module):
    def __init__(self, args, weights):
        '''
        args - 参数
        weights - 预训练权重
        is_att - 是否使用attention
        '''
        super(CNNInception, self).__init__()
        # 参数设定
        # 词汇表大小
        self.num_word = args.num_word
        # 词向量长度
        self.dim = args.dim
        # 类别数量
        self.num_class = args.num_class
        # 输入channel
        self.num_chan = args.num_chan
        # 卷积核的有关参数
        self.num_kerl = args.num_kerl
        self.kerl_sizes = args.kerl_sizes
        self.inception_size = args.inception_size

        # 定义各层
        # 不使用预训练词向量
        # self.embed = nn.Embedding(self.num_word, self.dim)
        # 使用预训练词向量
        self.embed = nn.Embedding.from_pretrained(weights)
        # 训练过程中对权重进行微调
        self.embed.weight.requires_grad = True
        # 卷积层和线性层
        self.convs = nn.Sequential(
                    InceptionLayer(self.dim, self.inception_size),
                    nn.MaxPool1d(179)
                )
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Sequential(
            nn.Linear(
                self.inception_size,
                self.num_class
            ),
        )
        

    def forward(self, x):
        # 词嵌入
        x = self.embed(x)
        # 卷积操作
        x = self.convs(x.permute(0,2,1))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class CNNConfig:
    '''
    CNN模型参数设定
    '''
    def __init__(self):
        # 共有参数
        # 词汇表大小
        self.num_word = 4955
        # 词向量长度
        self.dim = 300
        # 类别数量
        self.num_class = 7
        # drop out
        self.dropout = 0.5

        # CNN有关参数
        # 输入channel
        self.num_chan = 1
        # 每种卷积核的数量
        self.num_kerl = 350
        self.kerl_sizes = [2, 4, 6, 8, 10]
        # 是否attention
        self.is_att = False
        # inception及隐层大小
        self.inception_size = 5000
        
        # 训练参数设定
        self.num_epoch = 100
        self.lr = 1e-4
        # batch的大小
        self.batch_size = 80
        # 随机种子
        self.seed = 56
        # 是否有cuda
        self.cuda = (
            True if torch.cuda.is_available()
            else False
        )
        # weight decay
        self.weight_decay = 3



class RNNConfig:
    '''
    RNN模型参数设定
    '''
    def __init__(self):
        # 共有参数
        # 词汇表大小
        self.num_word = 4955
        # 词向量长度
        self.dim = 300
        # 类别数量
        self.num_class = 7
        # drop out
        self.dropout = 0.5
        
        # RNN参数
        # 隐藏层大小
        self.hidden_size = 256
        # 层数
        self.num_layers = 2
        
        # 训练参数设定
        self.num_epoch = 100
        self.lr = 1e-3
        # batch的大小
        self.batch_size = 80
        # 随机种子
        self.seed = 56
        # 是否有cuda
        self.cuda = (
            True if torch.cuda.is_available()
            else False
        )
        # weight decay
        self.weight_decay = 0.5


class FastTextConfig:
    '''
    FastText参数设定
    '''
    def __init__(self):
        # 共有参数
        # 词汇表大小
        self.num_word = 4955
        # 词向量长度
        self.dim = 300
        # 类别数量
        self.num_class = 7
        # drop out
        self.dropout = 0.5
        
        # FastText参数
        # 隐藏层大小
        self.hidden_size = 128
        # N-Gram词表大小
        self.num_gram = 250007
        
        # 训练参数设定
        self.num_epoch = 100
        self.lr = 0.0001
        # batch的大小
        self.batch_size = 80
        # 随机种子
        self.seed = 56
        # 是否有cuda
        self.cuda = (
            True if torch.cuda.is_available()
            else False
        )
        # weight decay
        self.weight_decay = 3
