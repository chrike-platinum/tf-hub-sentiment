from fastai import *
from fastai.text import *
#from fastai.docs import *
import pandas as pd
import torch
import json
import pickle
import dill as dill


''''
print('1')
DATA_PATH='tmp_data'
print('2')
df = pd.read_csv('twitter_data_small.csv', header=0)
df.columns=[0,1]

df_train = pd.read_csv('train_imdb.csv', header=None)
df_valid = pd.read_csv('valid_imdb.csv', header=None)


print('df1',df)
df=pd.concat([df,df_train,df_valid])
df = df.sample(frac=1).reset_index(drop=True)


print('df2',df)

df[:20000].to_csv('tmp_data/train.csv',index=False,header=False)
df[20001:22000].to_csv('tmp_data/valid.csv',index=False,header=False)

df=df.head()
print('input',df)
print('3')
classes = ['negative','positive']#read_classes(DATA_PATH/'classes.txt')
print('classes',classes)
print('4')
train_ds = TextDataset.from_csv(DATA_PATH, name='train', classes=classes)
print('5')
data_lm = text_data_from_csv(Path(DATA_PATH), data_func=lm_data)
print('6')
data_clas = text_data_from_csv(Path(DATA_PATH), data_func=classifier_data, vocab=data_lm.train_ds.vocab)
'''

'''
df_train = pd.read_csv('/Users/christiaanleysen/Downloads/ulm-basenet-master/runs/1/classifier/train.csv', header=None)
df_valid = pd.read_csv('Users/christiaanleysen/Downloads/ulm-basenet-master/runs/1/classifier/valid.csv', header=None)




print('df2',df)

df[:20000].to_csv('tmp_data/train.csv',index=False,header=False)
df[20001:22000].to_csv('tmp_data/valid.csv',index=False,header=False)

df=df.head()
print('input',df)
print('3')
classes = ['neg','pos']#read_classes(DATA_PATH/'classes.txt')
'''



#ownload_wt103_model()
df_train = pd.read_csv('/Users/christiaanleysen/Downloads/ulm-basenet-master/runs/1/classifier/train.csv', header=None)
mymap = {'neg':0, 'pos':1}
df_train[0]=df_train[0].apply(lambda s: mymap.get(s) if s in mymap else s)
print(df_train[0].value_counts())
print(df_train.head())

df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train[:14000].to_csv('data/train.csv',header=None,index=False)
nr_train=len(df_train)
print('Number of training set examples:',nr_train)

df_valid = pd.read_csv('/Users/christiaanleysen/Downloads/ulm-basenet-master/runs/1/classifier/valid.csv', header=None)
mymap = {'neg':0, 'pos':1}
print(df_valid[0].value_counts())
df_valid[0]=df_valid[0].apply(lambda s: mymap.get(s) if s in mymap else s)
print(df_valid.head())

#df_valid = df_valid.sample(frac=1).reset_index(drop=True)
df_valid[:14000].to_csv('data/valid.csv',header=None,index=False)
print('Number of validation set examples:',len(df_valid))


classes=['neg','pos']

DATA_PATH='data'

print('4')
train_ds = TextDataset.from_csv(DATA_PATH, name='train', classes=classes)

print('train_ds',train_ds.ids)
print('5')
data_lm = TextLMDataBunch.from_csv(DATA_PATH)


with open('vocabulary.pkl', 'wb') as outfile:
    pickle.dump(data_lm.train_ds.vocab,outfile)

print('JSON DUMPED')


print('6')
data_clas = TextClasDataBunch.from_csv(DATA_PATH, vocab=data_lm.train_ds.vocab, bs=128,num_workers=0)

print('dataset',data_clas.train_dl.one_batch())



print('7')
print('downloading')
#download_wt103_model()
print('8')





learn = RNNLearner.language_model(data_lm, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=0.5)
#print('lm',learn.model)
learn.fit_one_cycle(1, 1e-2)
#print('9')
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)
#print('10')
torch.save(learn.model[0], '/Users/christiaanleysen/Documents/Projects/tfhub_tl/model_sentiment5_lm.pt')

learn.save_encoder('/Users/christiaanleysen/Documents/Projects/tfhub_tl/output_temp/ft_enc')
#print('11')

print('===================encoder loaded=============')

learn = RNNLearner.classifier(data_clas, drop_mult=0.5)
learn.load_encoder('/Users/christiaanleysen/Documents/Projects/tfhub_tl/output_temp/ft_enc')

#learn.model.load_state_dict(torch.load(classifier_filename, map_location='cpu'))
print('RNN----------------------')

learn.fit_one_cycle(4, 1e-2)

learn.model.eval()
print('saving model')





torch.save(learn.model,'/Users/christiaanleysen/Documents/Projects/tfhub_tl/model_sentiment5.pt', pickle_module=dill)

print('model saved')

'''
print('12')
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
print('13')
learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))



learn.save('/Users/christiaanleysen/Documents/Projects/tfhub_tl/output/model_christiaan2')
learn.model.eval
learn.save('/Users/christiaanleysen/Documents/Projects/tfhub_tl/output/model_christiaan3')
learn.model.eval()
learn.save('/Users/christiaanleysen/Documents/Projects/tfhub_tl/output/model_christiaan4')
print(learn.model.eval())

'''

#(nr_train/2000)


