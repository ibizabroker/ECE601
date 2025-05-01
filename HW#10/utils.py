import torch
import torch.nn.functional as F
import numpy as np
import math
import os

#############################
# PTB
#############################

# Use this as the base path for all datasets
BASE_DATA_PATH = '/content/drive/MyDrive/ECE601-Module10/'

def check_ptb_dataset_exists(path_data=BASE_DATA_PATH):
    ptb_path = os.path.join(path_data, 'ptb')
    raw_data_path = os.path.join(ptb_path, 'data_raw')

    idx2word_path = os.path.join(ptb_path, 'idx2word.pt')
    test_data_path = os.path.join(ptb_path, 'test_data.pt')
    train_data_path = os.path.join(ptb_path, 'train_data.pt')
    word2idx_path = os.path.join(ptb_path, 'word2idx.pt')

    flag_idx2word = os.path.isfile(idx2word_path)
    flag_test_data = os.path.isfile(test_data_path)
    flag_train_data = os.path.isfile(train_data_path)
    flag_word2idx = os.path.isfile(word2idx_path)

    if not (flag_idx2word and flag_test_data and flag_train_data and flag_word2idx):
        print('PTB dataset missing - generating...')
        corpus = Corpus(raw_data_path)
        batch_size = 20
        train_data = batchify(corpus.train, batch_size)
        val_data = batchify(corpus.valid, batch_size)
        test_data = batchify(corpus.test, batch_size)
        vocab_size = len(corpus.dictionary)

        torch.save(train_data, train_data_path)
        torch.save(test_data, test_data_path)
        torch.save(corpus.dictionary.idx2word, idx2word_path)
        torch.save(corpus.dictionary.word2idx, word2idx_path)

    return path_data


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), f"File not found: {path}"
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


# Load dictionary (only if needed later)
ptb_path = check_ptb_dataset_exists()
word2idx = torch.load(os.path.join(ptb_path, 'ptb/word2idx.pt'))
idx2word = torch.load(os.path.join(ptb_path, 'ptb/idx2word.pt'))

# Additional utility functions like normalize_gradient, show_next_word, etc. can follow...
def normalize_gradient(net):

    grad_norm_sq=0

    for p in net.parameters():
        #grad_norm_sq += p.grad.data.norm()**2
        if p.grad is not None:
            grad_norm_sq += p.grad.data.norm()**2

    grad_norm=math.sqrt(grad_norm_sq)
   
    if grad_norm<1e-4:
        net.zero_grad()
        print('grad norm close to zero')
    else:    
        for p in net.parameters():
            if p.grad is not None:
                p.grad.data.div_(grad_norm)

    return grad_norm


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
         )

def sentence2vector(sentence):
    words = sentence.split()
    x = torch.LongTensor(len(words),1)
    for idx, word in enumerate(words):

         if word not in word2idx:
            print('You entered a word which is not in the vocabulary.')
            print('Make sure that you do not have any capital letters')
         else:
            x[idx,0]=word2idx[word]
    return x


def show_next_word(scores):
    num_word_display=30
    prob=F.softmax(scores,dim=2)
    p=prob[-1].squeeze()
    p,word_idx = torch.topk(p,num_word_display)

    for i,idx in enumerate(word_idx):
        percentage= p[i].item()*100
        word=  idx2word[idx.item()]
        print(  "{:.1f}%\t".format(percentage),  word ) 


