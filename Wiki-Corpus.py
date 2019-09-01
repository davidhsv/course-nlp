#!/usr/bin/env python
# coding: utf-8

# ## Database of all Spanish Wikipedia articles with >10000 words

# In[7]:
import html
import json
import re

import numpy as np

import pandas as pd

import pathlib
DATA_PATH = pathlib.Path.cwd()

# In[]:
jsons = []
for direc in (DATA_PATH/'./ptwiki').iterdir():
    print(direc)
    for file in direc.iterdir():
        print(file)
        for line in open(file, 'r', encoding="utf-8"):
            jsons.append(json.loads(line))

# In[4]:


jsons[0].keys()

# In[5]:


ID = 'id'
TITLE = 'title'
TEXT = 'text'
URL = 'url'

# In[6]:


idx = np.random.permutation(len(jsons))

# In[7]:


limit_sets = int(len(jsons) * 0.99)

trn_set = [jsons[i] for i in idx[:limit_sets]]
val_set = [jsons[i] for i in idx[limit_sets:]]


# In[8]:


def remove_title(texts):
    return texts.split('\n\n', 1)[-1]


# In[9]:


trn_ids = [exmp[ID] for exmp in trn_set]
trn_titles = [exmp[TITLE] for exmp in trn_set]
trn_texts = [remove_title(exmp[TEXT]) for exmp in trn_set]
trn_urls = [exmp[URL] for exmp in trn_set]

# In[10]:


val_ids = [exmp[ID] for exmp in val_set]
val_titles = [exmp[TITLE] for exmp in val_set]
val_texts = [remove_title(exmp[TEXT]) for exmp in val_set]
val_urls = [exmp[URL] for exmp in val_set]

# In[34]:


col_names = ['id', 'title', 'text', 'url']

# In[35]:


df_trn = pd.DataFrame({'id': trn_ids, 'title': trn_titles, 'text': trn_texts, 'url': trn_urls}, columns=col_names)
df_val = pd.DataFrame({'id': val_ids, 'title': val_titles, 'text': val_texts, 'url': val_urls}, columns=col_names)

# In[36]:


df_trn.head(5)

# In[37]:


# len(trn_set), len(val_set)


# In[38]:


# df_trn.to_csv(COMPL_PATH/'train+15.csv', header=False, index=False)
# df_val.to_csv(COMPL_PATH/'val+15.csv', header=False, index=False)


# In[11]:


col_names = ['title', 'text']

# In[12]:


df_trn = pd.DataFrame({'title': trn_titles, 'text': trn_texts}, columns=col_names)
df_val = pd.DataFrame({'title': val_titles, 'text': val_texts}, columns=col_names)

# In[14]:


df_trn.to_csv(DATA_PATH / 'train+100.csv', header=False, index=False)
df_val.to_csv(DATA_PATH / 'val+100.csv', header=False, index=False)

# ## Wiki model tokens

# In[3]:


chunksize = 4000

# In[4]:


re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ').replace('\xa0', ' ')
    return re1.sub(' ', html.unescape(x))


# In[5]:


def get_texts(df, n_lbls=0):
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls + 1, len(df.columns)):
        texts += f' {FLD} {i - n_lbls} ' + df[i].astype(str)
    texts = texts.apply(fixup).values.astype(str)

    tok = Tokenizer(lang='es').proc_all_mp(partition_by_cores(texts), lang='es')
    return tok


# In[6]:


def get_all(df, n_lbls):
    tok, labels = [], []
    l = 0
    for i, r in enumerate(df):
        print(i)
        l += 1
        tok_ = get_texts(r, n_lbls)
        tok += tok_;
        # if i >= 35:
        # return tok
    return tok


# In[8]:


df_trn = pd.read_csv(LM_PATH / 'train+100.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(LM_PATH / 'val+100.csv', header=None, chunksize=chunksize)

# In[20]:


(LM_PATH / 'tmp').mkdir(exist_ok=True)

# In[40]:


l = 0
for i, _ in enumerate(df_trn):
    l += 1
print(l)

# In[9]:


tok_val = get_all(df_val, 0)
np.save(LM_PATH / 'tmp' / 'tok_val+100.npy', tok_val)
tok_trn = get_all(df_trn, 0)
np.save(LM_PATH / 'tmp' / 'tok_trn+100.npy', tok_trn)

# In[11]:


from math import ceil


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


mini_chunks = chunks(tok_trn, ceil(len(tok_trn) / 4))
for i, mini in enumerate(mini_chunks):
    name = 'tmp/tok_trn' + str(i) + '+60.npy'
    np.save(LM_PATH / name, mini)

# In[7]:


# tok_trn0 = np.load(LM_PATH/'tmp'/'tok_trn0+100.npy')
# tok_trn1 = np.load(LM_PATH/'tmp'/'tok_trn1+100.npy')
# tok_trn2 = np.load(LM_PATH/'tmp'/'tok_trn2+100.npy')
# tok_trn3 = np.load(LM_PATH/'tmp'/'tok_trn3+100.npy')

# tok_trn = np.concatenate((tok_trn0, tok_trn1, tok_trn2, tok_trn3))
tok_trn = np.load(LM_PATH / 'tmp' / 'tok_trn+100.npy')
tok_val = np.load(LM_PATH / 'tmp' / 'tok_val+100.npy')

# We remove 3/8 of the total number of articles so that we keep ~120M tokens

# In[8]:


tok_trn = tok_trn[:(len(tok_trn) * 5) // 8]

# In[10]:


i = 0
for text in tok_trn:
    for char in text:
        i += 1
print(i)

# In[9]:


np.save(LM_PATH / 'tmp' / 'tok_trn120+100.npy', tok_trn)
tok_trn = np.load(LM_PATH / 'tmp' / 'tok_trn120+100.npy')

# In[12]:


freq = Counter(p for o in tok_trn for p in o)
freq.most_common(25)

# In[13]:


max_vocab = 60000
min_freq = 5

# In[14]:


itos2 = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
itos2.insert(0, '_pad_')
itos2.insert(0, '_unk_')
stoi2 = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos2)})
len(itos2)

# In[15]:


trn_lm = np.array([[stoi2[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi2[o] for o in p] for p in tok_val])

# In[16]:


np.save(LM_PATH / 'tmp' / 'trn_ids+100.npy', trn_lm)
np.save(LM_PATH / 'tmp' / 'val_ids+100.npy', val_lm)
pickle.dump(itos2, open(LM_PATH / 'tmp' / 'itos+100.pkl', 'wb'))

# ## Wiki+100 Model

# In[9]:


wd = 1e-7
bptt = 70
bs = 32
em_sz, nh, nl = 400, 1150, 3
# opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
opt_fn = partial(optim.SGD, momentum=0.9)

# In[10]:


trn_lm = np.load(LM_PATH / 'tmp' / 'trn_ids+100.npy')
val_lm = np.load(LM_PATH / 'tmp' / 'val_ids+100.npy')
trn_lm = np.concatenate(trn_lm)
val_lm = np.concatenate(val_lm)

# In[11]:


itos = pickle.load(open(LM_PATH / 'tmp' / 'itos+100.pkl', 'rb'))
vs = len(itos)

trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
val_dl = LanguageModelLoader(val_lm, bs, bptt)
md = LanguageModelData(MODEL_PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

# In[12]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.5

# In[13]:


learner = md.get_model(opt_fn, em_sz, nh, nl,
                       dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.unfreeze()

# In[12]:


learner.lr_find2(start_lr=1, end_lr=10, num_it=800)

# In[13]:


learner.sched.plot()

# In[40]:


learner.sched.plot()

# In[14]:


get_ipython().run_cell_magic('javascript', '', 'Jupyter.beep();')

# In[15]:


lr = 2
lrs = lr

# In[16]:


# learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)
learner.fit(lr, 1, cycle_len=10, use_clr_beta=(10, 10, 0.95, 0.85), best_save_name='first_run_portuguese')

# In[19]:


##learner.save('first_run_spanish')

# In[23]:


path = learner.models_path;
path

# ## Save Encoder and Weights

# In[16]:


learner.save_encoder('lm1_enc')

# In[17]:


learner.save('lm1_weights')
