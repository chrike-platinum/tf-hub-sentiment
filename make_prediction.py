from fastai import *
from fastai.text import *
import torch
import numpy as np
import pickle
import os


def create_dl(X):
    bs=64
    tst_ds = TextDataset([X], np.zeros(len(X)))
    tst_dl = DataLoader(tst_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=None)
    return tst_dl

def classifier_model_network(dir_path, cuda_id=0,dropmult=1.0):
    '''
    :param dir_path:
    :param cuda_id:
    :return:
    '''
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    dir_path = Path(dir_path)
    # load vocabulary lookup
    itos = pickle.load(open(dir_path / 'itos.pkl', 'rb'))
    n_tokens = len(itos)

    bptt = 70  # back propogation through time
    em_sz = 400  # size of embeddings
    nh = 1150  # size of hidden
    nl = 3  # number of layers
    label_class=['neg','pos']

    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * dropmult

    #m = get_rnn_classifier(bptt, 20 * 70, label_class, n_tokens, emb_sz=em_sz, n_hid=nh, n_layers=nl,
    #                       pad_token=1,
    #                       layers=[em_sz * 3, 50, label_class], drops=[dps[4], 0.1],
    #        dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    m=get_rnn_classifier(bptt, 20*70, len(label_class), n_tokens, em_sz, nh, nl, pad_token=1,
                      layers=[em_sz * 3, 50, len(label_class)], drops=[dps[4], 0.1], input_p=dps[0], weight_p=dps[1], embed_p=dps[2], hidden_p=dps[3])
    print('DONE')
    m.eval()  # just to make sure dropout is being applied

    #loaded_weights = torch.load(os.path.join(dir_path, "model_christiaan.h5.pth"), map_location='cpu')
    #m.load_state_dict(loaded_weights)
    #learner.load_encoder(os.path.join(dir_path, "model_christiaan.h5.pth"))

    return m






def get_learner2(dir_path, model_network, modelData, cuda_id=0):
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)
    if cuda_id == -1:
        map_location = 'cpu'
    else:
        map_location = None

    dir_path = Path(dir_path)

    learner = RNNLearner(modelData, TextModel(to_gpu(model_network)))
    learner.model.eval  # just to make sure dropout is being applied

    loaded_weights = torch.load(os.path.join(dir_path, "model_christiaan.h5.pth"), map_location=map_location)
    learner.load("fwd_clas_1")

    # confirmed that the new parameters match those of the loaded model
    for k, v in loaded_weights.items():
        print(k, np.all(v == learner.model.state_dict()[k]))

    return learner


def predict(learner: Learner, X):
    return [softmax(x) for x in learner.predict_dl(create_dl(X))]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_learner(dir_path, cuda_id):
    network = classifier_model_network(dir_path, cuda_id)
    learn = get_learner2(dir_path, network,cuda_id)
    return learn


# fetch the ids of the text
def get_tokens(sentence, lang='en'):
    '''
    fetch work tokens for the sentence
    :param text:
    :param lang:
    :return:
    '''
    tok = SpacyTokenizer('en')
    text = f'\n{BOS} {FLD} 1 ' + sentence
    return Tokenizer(lang=lang).process_text(text,tok)

def convert2ids(tokens: list, tok2id_model_path):
    itos = pickle.load(open(tok2id_model_path,'rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})

    predict_lm = np.array([stoi[p]  for p in tokens])
    return predict_lm

dir_path='/Users/christiaanleysen/Documents/Projects/tfhub_tl/output/'  # data directory path of the model built and saved .
lm_model_path= '/Users/christiaanleysen/Documents/Projects/tfhub_tl/output/itos.pkl' #file path of itos.pkl file in data_dir where you have tok2id file dumped.
toks = get_tokens('some text sentence you want to predict')
ids = convert2ids(toks, lm_model_path)


learner=get_learner(dir_path,cuda_id=-1)
predict(learner, (ids))