# For training the model
# created by wei
# Nov 8, 2022
# The tutorial page:
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/
# LUKE/Supervised_relation_extraction_with_LukeForEntityPairClassification.ipynb#scrollTo=EhOUh0qzFFje

from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from model import LUKE
import torch
from data_loader import label2id, train_dataloader, valid_dataloader, test_dataloader

model = LUKE(len(label2id))
print("--------------------model initialzation--------------------")
print("--------------------start training--------------------")
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=2,
    strict=False,
    verbose=False,
    mode='min'
)

# set max_epochs
trainer = Trainer(gpus=1, max_epochs=1,callbacks=[EarlyStopping(monitor='validation_loss')])
trainer.fit(model, train_dataloader, valid_dataloader)

# Evaluation
trainer.test()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

predictions_total = []
labels_total = []
for batch in tqdm(test_dataloader):
    # get the inputs;
    labels = batch["label"]
    del batch["label"]

    # move everything to the GPU
    for k,v in batch.items():
        batch[k] = batch[k].to(device)

    # forward pass
    outputs = model(**batch)
    logits = outputs.logits
    predictions = logits.argmax(-1)
    predictions_total.extend(predictions.tolist())
    labels_total.extend(labels.tolist())
print("Accuracy on test set:", accuracy_score(labels_total, predictions_total))
print("-----------------Training is end, start to save the model-----------------")
torch.save(model,'save.pth')
