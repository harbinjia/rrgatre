import os
import json

import torch
from torch._C import dtype
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer

from data_loader_utils import read_examples, convert_examples_to_features


class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

class CustomDataLoader(object):
    def __init__(self, params):
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size
        # self.parser = params.parser
        self.data_dir = params.data_dir
        self.rel2idx = params.rel2idx
        self.max_seq_length = params.max_seq_length
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),
                                       do_lower_case=False)
        self.data_cache = params.data_cache

    @staticmethod
    def collate_fn_train(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        edge_matrix = torch.tensor([f.edge_matrix for f in features], dtype=torch.long)
        seq_tags = torch.tensor([f.seq_tag for f in features], dtype=torch.long)
        poten_relations = torch.tensor([f.relation for f in features], dtype=torch.long)
        rel_tags = torch.tensor([f.rel_tag for f in features], dtype=torch.long)
        # corres_tags = torch.tensor([f.corres_tag for f in features], dtype=torch.long)
        
        subs = [f.subs for f in features]
        objs = [f.objs for f in features]
        tensors = [input_ids, attention_mask, edge_matrix, seq_tags, poten_relations, rel_tags, subs, objs]
        return tensors

    @staticmethod
    def collate_fn_test(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        edge_matrix = torch.tensor([f.edge_matrix for f in features], dtype=torch.long)
        input_tokens = [f.input_tokens for f in features]
        triples = [f.triples for f in features]
        tensors = [input_ids, attention_mask, edge_matrix, triples, input_tokens]
        return tensors

    def get_features(self, data_sign):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        """
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))
        # get features
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        if os.path.exists(cache_path) and self.data_cache:
          features = torch.load(cache_path)
        else:
          # get examples
          if data_sign in ("train", "val", "test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
              examples = read_examples(self.data_dir, data_sign=data_sign, rel2idx=self.rel2idx)
          else:
              raise ValueError("please notice that the data can only be train/val/test!!")
          features = convert_examples_to_features(self.params, examples, self.tokenizer, data_sign)
          # save data
          if self.data_cache:
              torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train"):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        """
        # InputExamples to InputFeatures
        features = self.get_features(data_sign=data_sign)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn_train)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn_test)
        elif data_sign in ("test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn_test)
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")
        return dataloader


if __name__ == '__main__':
    from utils import Params

    params = Params()
    dataloader = CustomDataLoader(params)
    feats = dataloader.get_features(data_sign='train')
    print(feats[7].subs)