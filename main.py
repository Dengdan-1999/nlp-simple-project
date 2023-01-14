
import paddle
import numpy as np
import paddlenlp
import pandas as pd
import re
from functools import partial
from paddlenlp.transformers import LinearDecayWithWarmup

from paddlenlp.data import Stack,Tuple,Pad
import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')
import paddle.nn.functional as F
# from utils import evaluate
from paddlenlp.datasets import MapDataset

# Train_path = '/home/dengdan/data/test_data.csv'             # 远程服务器上面的绝对路径
# Test_path='/home/dengdan/data/test_data.csv'

Train_path = './data/data103654/train.txt'             # 远程服务器上面的绝对路径
Test_path='./data/data103654/test.txt'
Valid_path='./data/data103654/dev.txt'

Label_list=['教育', '社会', '星座', '房产', '彩票', '体育', '游戏', '财经', '股票', '时政', '科技', '娱乐', '时尚', '家居']
id2label_list=[{0: '教育'}, {1: '社会'}, {2: '星座'}, {3: '房产'}, {4: '彩票'}, {5: '体育'}, {6: '游戏'},
               {7: '财经'}, {8: '股票'}, {9: '时政'}, {10: '科技'}, {11: '娱乐'}, {12: '时尚'}, {13: '家居'}]


class MyDataSet(paddle.io.Dataset):
    def __init__(self,path):
        self.data_dict_list=self.__load_data(path)


    def __load_data(self,path):
        with open(path,'r',encoding='utf-8') as f:
            data=f.readlines()
            data_dict_list=[]
            # label_dict_list=[]
            for line in data:
                message=line.strip('\n')
                label=message[-2:]
                for i in range(len(Label_list)):
                    if Label_list[i]==label:
                        label=i
                text=message[:-3]
                temp={'text':text,'label':label}
                data_dict_list.append(temp)
        return data_dict_list

    def __getitem__(self, idx):
        return self.data_dict_list[idx]

    def __len__(self):
        return len(self.data_dict_list)


batch_size = 32
max_seq_len = 64
lr = 5e-5
epochs = 1
MODEL_NAME = 'ernie-tiny'

# tokenizer作用为将原始输入文本转化成模型model可以接受的输入数据形式。
tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)


# 该函数作用为将数据转换为ernie所需要的输入数据格式
def convert_example(example, tokenizer, max_seq_length=128, is_test=False):
    encoded_inputs = tokenizer(text=example['text'], max_seq_len=max_seq_length)
    input_ids = encoded_inputs['input_ids']
    token_type_ids = encoded_inputs['token_type_ids']

    if is_test:
        print('进入了test:')
        return input_ids, token_type_ids
    else:
        label = np.array([example['label']], dtype='int64')
        return input_ids, token_type_ids, label


def create_dataloader(dataset, batch_size=1,
                      batchify_fn=None, trans_fn=None):
    # trans_fn对应前边的covert_example函数，使用该函数处理每个样本为期望的格式
    if trans_fn:
        dataset = dataset.map(trans_fn)

    # 定义并初始化数据读取器
    return paddle.io.DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=batchify_fn,
                                num_workers=1, drop_last=False, return_list=True)


trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_len)

trans_func02 = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_len,
    is_test=True)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    Stack(dtype='int64')
): [data for data in fn(samples)]

train_dataset = MyDataSet(Train_path)

train_set = MapDataset(train_dataset)

train_data_loader = create_dataloader(
    train_set,
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func
)


# 检测是否可以使用gpu,如果可以优先使用gpu
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device('gpu:0')

dropout_rate = None

# 学习率预热比例
warmup_propotion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs

lr_scheduler = LinearDecayWithWarmup(lr, num_training_steps, warmup_propotion)

# 构建网络
class ErnieForSequenceClassification(paddle.nn.Layer):
    def __init__(self, num_class=14, dropout=None):
        super(ErnieForSequenceClassification, self).__init__()
        # 加载预训练好的ernie
        self.ernie = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
        # self.dropout=paddle.nn.Dropout(dropout)
        self.dropout = paddle.nn.Dropout(dropout if dropout is not None else self.ernie.config['hidden_dropout_prob'])
        self.classifier = paddle.nn.Linear(self.ernie.config['hidden_size'], num_class)

    def forward(self, input_ids, token_type_ids=None):
        sequence_output, pooled_output = self.ernie(
            input_ids,
            token_type_ids)
        # print("在网络里面~")
        # print('pooled_output:{}'.format(pooled_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# 加载预训练模型ERNIE
# 加载用于文本分类的fune-tuning网络
model = ErnieForSequenceClassification(num_class=14,
                                       dropout=dropout_rate)

# 定义统计指标
metric = paddle.metric.Accuracy()

optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'norm'])
    ])

def evaluate(model, metric, data_loader):
    model.eval()
    # 每次使用测试集进行评估时，先重置掉之前的metric的累计数据，保证只是针对本次评估。
    metric.reset()
    losses = []
    # logits_list=[]
    # labels_list=[]
    acc_list = []
    total_acc = 0

    for step, batch in enumerate(data_loader):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        # logits_list.extend(logits)
        # labels_list.extend(labels)
        loss = F.cross_entropy(input=logits, label=labels)
        loss = paddle.mean(loss)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        acc = metric.accumulate()
        acc_list.append(acc)
        total_acc += acc

    print('eval loss:%.5f,acc:%.5f' % (np.mean(losses), acc))
    print('acc_list={}'.format(acc_list))
    print('ave_acc={}'.format(np.mean(acc_list)))
    print('total_acc={},ave_acc={}'.format(total_acc, total_acc / len(acc_list)))
    metric.reset()


def train(model):
    global_step = 0
    for epoch in range(1, epochs + 1):
        print('epoch={}'.format(epoch))
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):
            if (step == 100):
                print("step={}".format(step))
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)

            loss = F.cross_entropy(input=logits, label=labels)

            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            # if step<5:
            #     print('probs.shape:{}'.format(probs.shape))
            #     pred_labels=np.argmax(probs,axis=1)
            #     true_labels=labels
            #     print('step:{},pred_labels:{},true_labels:{}'.format(step,pred_labels,true_labels))
            #     # print('step:{},correct:{}'.format(step,correct))

            metric.update(correct)
            acc = metric.accumulate()
            print('step:{},acc:{}'.format(step, acc))

            global_step += 1

            # print("global step %d,epoch:%d,batch:%d,loss:%.5f,acc:%.5f"%(
            #     global_step,epoch,step,loss,acc))
            # if global_step % 10==0:
            #     print('batch:{}'.format(batch))
            # print('probs:{}\nlabels={}'.format(probs,labels))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        # evaluate(model,metric,test_data_loader)

train(model)

# 模型保存的名称
model_name = "ernie_for_news_classification"

paddle.save(model.state_dict(), "{}.pdparams".format(model_name))
paddle.save(optimizer.state_dict(), "{}.optparams".format(model_name))
tokenizer.save_pretrained('./tokenizer')


valid_dataset = MyDataSet(Valid_path)

valid_set = MapDataset(valid_dataset)
# test_set = MapDataset(test_dataset)

valid_data_loader = create_dataloader(
    valid_set,
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func
)

evaluate(model, metric,valid_data_loader)

# ------------------------------------   保存模型用于预测（推理）------------------------------------
from paddle.static import InputSpec
# 1.切换 eval()模式
model.eval()
# 2. 构造 InputSpec 信息
input_ids = InputSpec([32, 26],'int64')
segment_ids=InputSpec([32,26],'int64')
# 3.调用 paddle.jit.save 接口转为静态图模型
path = "model_for_predict/linear"
paddle.jit.save(
    layer=model,
    path=path,
    input_spec=[input_ids,segment_ids])

# ------------------------------------   加载模型用于预测 ------------------------------------
import paddle
path = "model_for_predict/linear"
loaded_model = paddle.jit.load(path)
# loaded_model.eval()

# ------------------------------------   定义测试数据集读取器 ------------------------------------
class TestDataSet(paddle.io.Dataset):
    def __init__(self,path):
        self.data_dict_list=self.__load_data(path)


    def __load_data(self,path):
        with open(path,'r',encoding='utf-8') as f:
            data=f.readlines()
            data_dict_list=[]
            for line in data:
                message=line.strip('\n')
                text=message
                temp={'text':text}
                data_dict_list.append(temp)
        return data_dict_list[:1000]

    def __getitem__(self, idx):
        return self.data_dict_list[idx]

    def __len__(self):
        return len(self.data_dict_list)

# tokenizer作用为将原始输入文本转化成模型model可以接受的输入数据形式。
import paddlenlp
from functools import partial
from paddlenlp.data import Stack,Tuple,Pad
N=0
max_seq_len=64


tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained('ernie-tiny')
# 该函数作用为将数据转换为ernie所需要的输入数据格式

def convert_example02(example, tokenizer, max_seq_length=128):
    # global N
    # N+=1
    encoded_inputs = tokenizer(text=example['text'], max_seq_len=max_seq_length)
    input_ids = encoded_inputs['input_ids']
    token_type_ids = encoded_inputs['token_type_ids']
    # print('{},进入了test:'.format(N))
    return input_ids, token_type_ids


def create_dataloader02(dataset, batch_size=1,
                      batchify_fn=None, trans_fn=None):
    # trans_fn对应前边的covert_example函数，使用该函数处理每个样本为期望的格式
    if trans_fn:
        dataset = dataset.map(trans_fn)
    print('进入了create_dataloader02')

    # 定义并初始化数据读取器
    return paddle.io.DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=batchify_fn,
                                num_workers=1, drop_last=False, return_list=True)


# trans_func = partial(
#     convert_example,
#     tokenizer=tokenizer,
#     max_seq_length=max_seq_len)

trans_func02 = partial(
    convert_example02,
    tokenizer=tokenizer,
    max_seq_length=max_seq_len)

batchify_fn02 = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
): [data for data in fn(samples)]

Test_path='./data/data103654/test.txt'
from paddlenlp.datasets import MapDataset
batch_size=32

# 加载测试数据集
test_dataset=TestDataSet(Test_path)
test_set = MapDataset(test_dataset)
test_data_loader = create_dataloader02(
    test_set,
    batch_size=batch_size,
    batchify_fn=batchify_fn02,
    trans_fn=trans_func02
)

# 将该模型及其所有子层设置为预测模式
loaded_model.eval()

# ------------------------------------   定义预测函数并预测 ------------------------------------
def predict(model,data_loader):
    model.eval()
    results = []

    for step, batch in enumerate(data_loader):
        input_ids, segment_ids = batch
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        results.extend(idx)
    return results

pred_label_id=predict(loaded_model,test_data_loader)
pred_label=[]
for i in range(1000):
    n=pred_label_id[i]
    label=Label_list[n]
    print('label=',label)
    print('n={},label={}'.format(n,label))
    pred_label.append(label)
