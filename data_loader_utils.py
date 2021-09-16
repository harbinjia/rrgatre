import pandas as pd
from time import time
import logging
import json
from collections import defaultdict
# import spacy
# parser = spacy.load('en_core_web_md')
import numpy as np
from itertools import chain
from multiprocessing import Pool
import functools

Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}

class InputExample(object):
  """a single set of samples of data
  """

  def __init__(self, text, tokens, edges, en_pair_list, re_list, rel2ens):
      self.text = text
      self.tokens = tokens
      self.edges = edges
      self.en_pair_list = en_pair_list
      self.re_list = re_list
      self.rel2ens = rel2ens

def read_examples(data_dir, data_sign, rel2idx):
    """load data to InputExamples
    """
    examples = []
    # rel2idx = params.rel2idx
    # read src data
    with open(data_dir / f'{data_sign}.json', "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
          text = sample['text']
          tokens = sample['tokens']
          edges = sample['edges']
          rel2ens = defaultdict(list)
          en_pair_list = []
          re_list = []
          for triple in sample['triples']:
              en_pair_list.append([triple[0], triple[-1]])
              re_list.append(rel2idx[triple[1]])
              rel2ens[rel2idx[triple[1]]].append((triple[0], triple[-1]))
          example = InputExample(text=text, tokens=tokens, edges=edges, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
          examples.append(example)
    print("InputExamples:", len(examples))
    return examples

class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
          input_ids,
          attention_mask,
          edge_matrix,
          input_tokens=None,
          seq_tag=None,
          triples=None,
          relation=None,
          rel_tag=None,
          # corres_tag=None,
          subs = None,
          objs = None
          ):
      
      self.input_tokens = input_tokens
      self.input_ids = input_ids
      self.attention_mask = attention_mask
      self.edge_matrix = edge_matrix 
      self.seq_tag = seq_tag
      self.triples = triples
      self.relation = relation
      self.rel_tag = rel_tag
      # self.corres_tag = corres_tag
      self.subs  = subs
      self.objs = objs

def create_graph_from_sentence_and_word_vectors(matrix_len, edges):
    """
    construct graph for depdency edge, entity
    """
    # matrix_len = len(words)
    A_fw = np.zeros(shape=(matrix_len, matrix_len))
    # A_bw = np.zeros(shape=(matrix_len, matrix_len))

    for i in range(matrix_len):
        A_fw[i][i] = 1
        # A_bw[i][i] = 1

    for (word1,word2) in edges:
        if word1 >= matrix_len or word2 >= matrix_len:
            continue
        else:
            A_fw[word1][word2] = 1
            A_fw[word2][word1] = 1

    return A_fw

def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def _get_so_head(en_pair, tokenizer, text_tokens):
  sub = tokenizer.tokenize(en_pair[0])
  obj = tokenizer.tokenize(en_pair[1])
  sub_head = find_head_idx(source=text_tokens, target=sub)
  if sub == obj:
    obj_head = find_head_idx(source=text_tokens[sub_head + len(sub):], target=obj)
    if obj_head != -1:
      obj_head += sub_head + len(sub)
    else:
      obj_head = sub_head
  else:
    obj_head = find_head_idx(source=text_tokens, target=obj)
  # sub_t = (sub_head, sub_head + len(sub) - 1)
  # obj_t = (obj_head, obj_head + len(obj) - 1)
  return sub, obj, sub_head, obj_head

def change_edge(sp_token, sp_edges, tokenizer):
  # 将spacy dep_rel token id 转为word token id

  word_indexer = []
  tokens = []

  for word in sp_token:
      word_tokens = tokenizer.tokenize(word)
      token_idx = len(tokens)
      tokens.extend(word_tokens)
      # word_indexer is for indexing after bert, feature back to the length of original length.
      word_indexer.append(token_idx)
      # word_indexer = pad_sequence(word_indexer, batch_first=True, padding_value=1)
  word_indexer = [i+1 for i in word_indexer]
  edges = [[word_indexer[l[0]], word_indexer[l[1]]] for l in sp_edges]
  return edges

def convert(example, max_text_len, tokenizer, rel2idx, data_sign):

    """convert function
      params:
        max_text_len: bert tokenize max_text_len
        tokenizer: bert tokenizer
        parser: spacy tokenizer
    """

    text_tokens = tokenizer.tokenize(example.text)  
    # cut off
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    edges = change_edge(example.tokens, example.edges, tokenizer)
    # A_fw: edge adj matrix
    A_fw = create_graph_from_sentence_and_word_vectors(max_text_len, edges)

    # train data
    if data_sign == 'train':
        
        rel_tag = len(rel2idx) * [0]
        corres_tag = np.zeros((max_text_len, max_text_len))

        subs = []  # save the sub idx (h, t)
        objs = []
        for en_pair, rel in zip(example.en_pair_list, example.re_list):
            sub, obj, sub_head, obj_head = _get_so_head(en_pair, tokenizer, text_tokens)
            rel_tag[rel] = 1
            subs.append((sub_head, sub_head + len(sub) - 1))
            objs.append((obj_head, obj_head + len(obj) - 1))
            # if sub_head != -1 and obj_head != -1:
            #     corres_tag[sub_head][obj_head] = 1

        sub_feats = []

        # positive samples
        for rel, en_ll in example.rel2ens.items():
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxSub['O']]
            for en in en_ll:
                subj, obj, sub_head, obj_head = _get_so_head(en, tokenizer, text_tokens)
                if sub_head != -1 and obj_head != -1:
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            seq_tag = [tags_sub, tags_obj]
            sub_feats.append(InputFeatures(
                # input_tokens=text_tokens,  # bert tokens 
                input_ids = input_ids,       # bert token to id
                attention_mask = attention_mask,  # mask token with 1, padding 0 to maxlen
                edge_matrix = A_fw,         # edge matrix with spacy dependency relation, shape [bs, 100, 100]
                # corres_tag = corres_tag,      # all entity head matrix of a text, shape [bs, 100, 100]
                seq_tag = seq_tag,         # list[sub, obj], label with BIO->120
                relation = rel,           # relation of a triple of a text, shape [bs, 23]
                rel_tag = rel_tag,         # all relation of a text, [bs, 23]
                subs = subs,
                objs = objs
            ))
    else:
        triples = []
        sub_feats = []
        for rel, en in zip(example.re_list, example.en_pair_list):
            # get sub and obj head
            sub, obj, sub_head, obj_head = _get_so_head(en, tokenizer, text_tokens)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub))
                t_chunk = ('T', obj_head, obj_head + len(obj))
                triples.append((h_chunk, t_chunk, rel))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples,
                edge_matrix=A_fw
            )
        ]
    return sub_feats
    

def convert_examples_to_features(params, examples, tokenizer, data_sign):
    """convert examples to features.
    :param examples (List[InputExamples])
    """
    max_text_len = params.max_seq_length
    rel2idx = params.rel2idx
    # multi-process
    with Pool(10) as p:
        convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
                                         data_sign=data_sign)
        features = p.map(func=convert_func, iterable=examples)

    return list(chain(*features))