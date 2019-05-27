
import  torch
from PIL import Image
from torchvision import *
from torch.autograd import Variable
from fastai.text import *
import numpy as np
import pickle

#input = torch.zeros(200, 20).long() + 1


def pad_collate(samples, pad_idx=1, pad_first=True):
    "Function that collect samples and adds padding."
    max_len = max([len(s) for s in samples])
    res = torch.zeros(max_len, len(samples)).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[-len(s):,i] = torch.LongTensor(s)
        else:         res[:len(s):,i] = torch.LongTensor(s)
    return res

def get_tokens(sentence, lang='en'):

    text = 'xxfld 1 ' + sentence #xxbos is not needed maybe
    return text
'''
def convert2ids(tokens: list, tok2id_model_path):
    itos = pickle.load(open(tok2id_model_path,'rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})

    predict_lm = np.array([stoi[p]  for p in tokens])
    return predict_lm

'''

tok=Tokenizer()
print('tokenizer initialized')



#tokens=tok.process_all(['The product was terrible.','the product was amazing']
tokens=tok.process_all([get_tokens(sentence) for sentence in ['The film was amazing, i loved all the characters. This is a real 10 out of 10.','worst movie of all time!! The actors were terrible!', 'the movie was amazing.']])


print(tokens)
#vocab = Vocab.create('/Users/christiaanleysen/Documents/Projects/tfhub_tl/data/tmp2/', tokens, max_vocab=1000, min_freq=2)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

with open('vocabulary.pkl','rb') as data_file:
    vocab = pickle.load(data_file)

print('vocab',vocab)
input=[vocab.numericalize(token) for token in tokens]


print('input',input)

input=pad_collate(input)






print('padded input',input)

#input = torch.tensor(input).long() + 1

mod=torch.load('model_sentiment4.pt')
#mod.reset()
mod.eval()
print('model loaded')

res=mod(Variable(input))[0].data.numpy()

print(res)


print([softmax(x) for x in res])

