from transformers import DetrForObjectDetection
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, AdamW
from datasets import load_dataset
from tqdm.auto import tqdm
from dataset import WiderFaceSet, collate_fn
from loss import Loss_fn
from args import batch_size, learning_rate, num_epochs, PATH, accelerator


model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model.class_labels_classifier = nn.Linear(256, 1, bias=True)
for p in model.parameters():
    p.requires_grad = True
for p in model.class_labels_classifier.parameters():
    if p.dim() > 1:
        nn.init.kaiming_uniform_(p)

wider_face = load_dataset('Wodeyuanbukongda/wider_face_no_face_central_coordinate')
wider_face = wider_face['train'].train_test_split(test_size=0.2)
dataset_train = WiderFaceSet(wider_face['train'])
dataset_val = WiderFaceSet(wider_face['test'])
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, collate_fn=collate_fn)

optim = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)
loss_fn = Loss_fn()
try:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
except:
    start_epoch = 0
optim, scheduler, dataloader_train = accelerator.prepare(optim, scheduler, dataloader_train)

count = 0
for epoch in range(start_epoch, num_epochs):
    progress_bar = tqdm(dataloader_train)
    model.train()
    model = accelerator.prepare(model)
    for batch in dataloader_train:
        with accelerator.accumulate(model):
            inputs = batch['inputs']
            bbox = batch['bbox']
            labels = batch['labels']
            outputs = model(**inputs)
            loss, iou = loss_fn(outputs['logits'], outputs['pred_boxes'], labels, bbox)

            optim.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optim.step()

        progress_bar.update(1)
    with accelerator.accumulate(model):
        scheduler.step()
    model = accelerator.unwrap_model(model)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, PATH)
    print('epoch: ', epoch, ' loss: ', loss.item())
