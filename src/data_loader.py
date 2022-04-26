import os
import re
import random
import numpy as np
from tqdm import tqdm_notebook
from mmsdk import mmdatasdk as md
from collections import defaultdict
from transformers import AutoTokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from create_dataset import MOSI, MOSEI, PAD, UNK

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
deberta_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

class MSADataset(Dataset):
    def __init__(self, config):
        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        else:
            print("Dataset not defined correctly")
            exit()

        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""
    dataset = MSADataset(config)
    if "mosi" in str(config.data_dir).lower():
        DATASET = md.cmu_mosi
    elif "mosei" in str(config.data_dir).lower():
        DATASET = md.cmu_mosei
    train_split = DATASET.standard_folds.standard_train_fold
    dev_split = DATASET.standard_folds.standard_valid_fold
    test_split = DATASET.standard_folds.standard_test_fold

    print(config.mode)
    config.data_len = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        # get raw text
        text_list = []
        if "mosi" in str(config.data_dir).lower():
            for sample in batch:
                segment = sample[2]
                pattern = re.compile('(.*)\[.*\]')
                vid = re.search(pattern, segment).group(1)
                idx = str(int(segment.replace(vid, '')[1:-1])+1)
                if vid in train_split:
                    split = 'train'
                elif vid in dev_split:
                    split = 'val'
                elif vid in test_split:
                    split = 'test'
                file = open(os.path.join('/home/ICT2000/yin/emnlp/data/CMU-MOSI/text', split, vid+'_'+idx+'.txt'), 'r')
                sentence = file.readline()
                text_list.append(sentence)
        elif "mosei" in str(config.data_dir).lower():
            for sample in batch:
                segment = sample[2]
                pattern = re.compile('(.*)\[.*\]')
                vid = re.search(pattern, segment).group(1)
                if vid in train_split:
                    split = 'train'
                elif vid in dev_split:
                    split = 'val'
                elif vid in test_split:
                    split = 'test'
                idx = int(segment.replace(vid, '')[1:-1])
                file_list = sorted(os.listdir(os.path.join('/home/ICT2000/yin/emnlp/data/CMU-MOSEI/text', split, vid)))
                file = open(os.path.join('/home/ICT2000/yin/emnlp/data/CMU-MOSEI/text', split, vid, file_list[idx]), 'r')
                sentence = file.readline()
                text_list.append(sentence)

        if config.text_encoder == 'bert':
            SENT_LEN = sentences.size(0)
            bert_details = []
            for sample in batch:
                text = " ".join(sample[0][3])
                encoded_bert_sent = bert_tokenizer.encode_plus(
                    text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True)
                bert_details.append(encoded_bert_sent)

            bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
            bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
        elif config.text_encoder == 'roberta':
            encoded_bert_sent = roberta_tokenizer(text_list, padding=True, truncation=True,
                                            max_length=roberta_tokenizer.model_max_length, return_tensors="pt")

            bert_sentences = torch.LongTensor(encoded_bert_sent["input_ids"])
            bert_sentence_types = bert_sentences
            bert_sentence_att_mask = torch.LongTensor(encoded_bert_sent["attention_mask"])
        elif config.text_encoder == 'deberta':
            encoded_bert_sent = deberta_tokenizer(text_list, padding=True, truncation=True,
                                            max_length=deberta_tokenizer.model_max_length, return_tensors="pt")

            bert_sentences = torch.LongTensor(encoded_bert_sent["input_ids"])
            bert_sentence_types = bert_sentences
            bert_sentence_att_mask = torch.LongTensor(encoded_bert_sent["attention_mask"])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
