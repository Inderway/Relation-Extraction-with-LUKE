# inference of the model
# created by wei Nov 8 2022

import torch
import torch.nn.functional as F
from data_loader import test_df, id2label, tokenizer
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model=torch.load("save.pth").to(device)
idx=random.randint(0, test_df.shape[0]-1)
text = test_df.iloc[idx].sentence
entity_spans = test_df.iloc[idx].entity_spans  # character-based entity spans 
entity_spans = [tuple(x) for x in entity_spans]
inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt").to(device)

outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Index is ", idx)
print("Sentence:\n", text)
print("Entities are: ")
for _, t in enumerate(entity_spans):
    print('--'+text[t[0]:t[1]])
print("Ground truth label:\n", test_df.iloc[idx].string_id)
print("Predicted class idx:\n", id2label[predicted_class_idx])
print("Confidence:\n", F.softmax(logits, -1).max().item())