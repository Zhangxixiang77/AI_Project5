import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import RobertaTokenizer


class MyDataset(Dataset):
    def __init__(self, args, data, transform=None):
        self.args = args
        self.data = data
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
        self.label_dict_number = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }
        self.label_dict_str = {
            0: 'negative',
            1: 'neutral',
            2: 'positive',
        }

    def __getitem__(self, index):
        return self.tokenize(self.data[index])

    def __len__(self):
        return len(self.data)

    def tokenize(self, item):
        item_id = item['id']
        text = item['text']
        img = item['img']
        label = item['label']

        text_token = self.tokenizer(text, return_tensors="pt", max_length=self.args.text_size,
                                    padding='max_length', truncation=True)
        text_token['input_ids'] = text_token['input_ids'].squeeze()
        text_token['attention_mask'] = text_token['attention_mask'].squeeze()

        img_token = self.transform(img) if self.transform else torch.tensor(img)

        label_token = self.label_dict_number[label] if label in self.label_dict_number else -1
        return item_id, text_token, img_token, label_token


def load_json(file):
    data_list = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = json.load(f)
        for line in lines:
            item = {
                'img': np.array(Image.open(line['img'])),
                'text': line['text'],
                'label': line['label'],
                'id': line['guid']
            }
            data_list.append(item)
    return data_list


def load_data(args):
    img_size = (args.img_size, args.img_size)
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(img_size),
         transforms.Normalize([0.5], [0.5])]
    )
    data_list = {
        'train': load_json(args.train_file),
        'dev': load_json(args.dev_file),
        'test': args.test_file and load_json(args.test_file),
    }
    data_set = {
        'train': MyDataset(args, data_list['train'], transform=data_transform),
        'dev': MyDataset(args, data_list['dev'], transform=data_transform),
        'test': args.test_file and MyDataset(args, data_list['test'], transform=data_transform),
    }
    return data_set[args.mode], data_set['dev']


def save_data(file, predict_list):
    with open(file, 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for pred in predict_list:
            f.write(f"{pred['guid']},{pred['tag']}\n")