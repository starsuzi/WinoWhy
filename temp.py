import random
import numpy
import sys
import torch
import argparse
import pandas as pd 

def output_five_folds(cls_df, method, shuffle,usage):
    """
    method in ["r","wsc"]
    shuffle in [True, False]
    usage in ["wnli", "cls"]
    
    """
    seed = 2019
    info = method + "_" + str(shuffle) + "_" + usage +"_" + str(seed)
    
    five_folds = []
    
    if not shuffle:
        working_df = cls_df
    elif shuffle:
        working_df = shuffle(cls_df,random_state=seed)
    
    if method == "r":
            raw_five_folds = np.array_split(working_df,5)
    elif method == "wsc":
        raw_five_folds = []
        for i in range(5):
             raw_five_folds.append(cls_df.loc[cls_df['fold_num'] == i])           
    
    for fold in raw_five_folds:
        if usage == "cls":
            five_folds.append(fold[["sentence","label"]])
        elif usage == "wnli":
            five_folds.append(fold[["wnli_sent1","wnli_sent2","label"]])

    print("working with classification data with examples:", cls_df.shape[0])

    return five_folds

class DataLoader:
    def __init__(self, data_path, args):
        self.args = args
        with open(data_path, 'r') as f:
            self.cls_df = pd.read_csv(f)

        self.five_folds = output_five_folds(self.cls_df, method=args.method, shuffle=False, usage="wnli")

        self.test_df,self.train_df = pick_training_and_testing_folds(self.five_folds, args.fold)

        self.train_set = self.tensorize_example(self.train_df)
        print('successfully loaded %d examples for training data' % len(self.train_set))

        self.test_set = self.tensorize_example(self.test_df)
        print('successfully loaded %d examples for test data' % len(self.test_set))

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = numpy.zeros(300)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = numpy.fromstring(line[word_end + 1:], numpy.float32, sep=" ")
                    assert len(embedding) == 300
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict

    def tensorize_example(self, initial_dataframe):
        tensorized_dataset = list()
        
        for i in range(initial_dataframe.shape[0]):

            tensorized_examples_for_one_frame = list()

            sent1 = initial_dataframe.iloc[i]["wnli_sent1"]
            sent2 = initial_dataframe.iloc[i]["wnli_sent2"]
            label = initial_dataframe.iloc[i]["label"]

            lm_tokenized_sent1 = tokenizer.encode(sent1)
            lm_tokenized_sent2 = tokenizer.encode(sent2)
            bert_tokenized_sent1 = tokenizer.encode('[CLS] ' + sent1 + ' . [SEP]')
            bert_tokenized_sent2 = tokenizer.encode('[CLS] ' + sent2 + ' . [SEP]')

            tensorized_examples_for_one_frame.append(
                {'gpt2_sent1':torch.tensor(lm_tokenized_sent1).to(device),
                    'gpt2_sent2': torch.tensor(lm_tokenized_sent2).to(device),
                    'bert_sent1':torch.tensor(bert_tokenized_sent1).to(device),
                    'bert_sent2': torch.tensor(bert_tokenized_sent2).to(device),
                    'label': torch.tensor([int(label)]).to(device)
                    })

            tensorized_dataset += tensorized_examples_for_one_frame

        return tensorized_dataset

parser = argparse.ArgumentParser()
args = parser.parse_args()
all_data = DataLoader('./dataset/dataset.csv', args)

'''
test = sys.argv[1]
with open("./commonsense_ability_test/{}.txt".format(test), "r") as f:
    file = f.readlines()
num = len(file)
count = 0
curr = 0

X = []
y = []

for line in file:
    lst = []
    line = line.strip().split("\001")
    label = int(line[0])
    #print(type(label))
    y.append(label)
    score_list = []
    for sentence in line[1:]:
        #i = i+1
        #print(i)
        #print(sentence)
        #print(type(sentence))
        lst.append(sentence)
    X.append(lst)

print()
#print(X[0:2])
#print(len(X[0:2]))
#print(y)

random.seed(0)
random.shuffle(X)
#print(len(X))
X_train = X[:int((len(X)+1)*.80)] #Remaining 80% to training set
y_train = y[:int((len(y)+1)*.80)]
X_test = X[int((len(X)+1)*.80):] #Splits 20% data to test set
y_test = y[int((len(y)+1)*.80):]
#print(indices)
#print(len(indices))
#print(len(X))
#print(len(X))
#print(len(y))

ds = TensorDataset(X, y)
loader = DataLoader(ds, shuffle=True)

#print(X_train)
#print(len(X_train))
#print(len(X_val))
#print(len(X_test))
'''