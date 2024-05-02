import torch
from torch.utils.data import Dataset
from transformers import DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


class WiderFaceSet(Dataset):
    def __init__(self, wider_face):
        super(WiderFaceSet, self).__init__()
        self.wider_face = wider_face

    def __getitem__(self, index):
        example = self.wider_face[index]
        image = example['image']
        labels = example['label']
        bbox = example['bbox']
        return {'image': image, 'labels': labels, 'bbox': bbox}

    def __len__(self):
        return len(self.wider_face)


def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float32)
    bbox = torch.tensor([item['bbox'] for item in batch], dtype=torch.float32)
    inputs = processor(images=images, return_tensors='pt')
    return {'inputs': inputs, 'labels': labels, 'bbox': bbox}

