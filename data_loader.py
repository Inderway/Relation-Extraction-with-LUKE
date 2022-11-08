# For preparing the dataset
# created by wei
# Nov 8, 2022

import requests, zipfile, io, os
import pandas as pd
from transformers import LukeTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

def download_data():
    url = "https://www.dropbox.com/s/izi2x4sjohpzoot/relation_extraction_dataset.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

if not os.path.exists("relation_extraction_dataset.pkl"):
    download_data()
print("--------------------Data has been prepared------------")

# print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

# a table with 12031 rows and 8 columns 
df = pd.read_pickle("relation_extraction_dataset.pkl")
df.reset_index(drop=True, inplace=True)

# create a dictionary that maps ids to labels
id2label = dict()
for idx, label in enumerate(df.string_id.value_counts().index):
    id2label[idx] = label

#recontruct id2label which maps each label to an idx
label2id = {v:k for k,v in id2label.items()}

# full print
torch.set_printoptions(profile="full")

# a tool for NER
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_pair_classification")

class RelationExtractionDataset(Dataset):
    """Relation extraction dataset."""

    def __init__(self, data):
        """
        Args:
            data : Pandas dataframe.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        sentence = item.sentence
        # two entities
        entity_spans = [tuple(x) for x in item.entity_spans]

        encoding = tokenizer(sentence, entity_spans=entity_spans, padding="max_length", truncation=True, return_tensors="pt")
        for k,v in encoding.items():
            encoding[k] = encoding[k].squeeze()
        encoding["label"] = torch.tensor(label2id[item.string_id])
        return encoding

# set datasets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=False)

# define the dataset
train_dataset = RelationExtractionDataset(data=train_df)
valid_dataset = RelationExtractionDataset(data=val_df)
test_dataset = RelationExtractionDataset(data=test_df)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2)
test_dataloader = DataLoader(test_dataset, batch_size=2)

print("--------------------Dataset done--------------------")

# An example of a batch
batch = next(iter(train_dataloader))
'''
batch is a dictionary
The keys of batch are as following:
-input_ids: the token ids sequence of a sentence
-entity_ids: the entity ids of the two entity in the sentence
-entity_position_ids: the position of the entities
-attention_mask
-entity_attention_mask
-label

The shape of the related values are 4x512, 4x2, 4x2x30, 4x512, 4x2, 4x1
'''
