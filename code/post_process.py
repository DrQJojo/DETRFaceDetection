import torch
from PIL import ImageDraw
from args import threshold


def xcycwh_to_x1y1x2y2(bboxes, size):
    # bboxes.shape = [N,4]
    xc, yc, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    height, width = size
    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height
    bboxes = torch.stack([x1, y1, x2, y2], dim=1)
    return bboxes


def draw_bbox(image, bboxes, scores):
    draw = ImageDraw.Draw(image)
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2, y2), outline="red", width=1)
        draw.text((x1, y1 - 15), 'score: ' + str(score), fill='green')
    return image


def post_process(image, logits, bboxes):
    prob = torch.sigmoid(logits).squeeze()
    bboxes = bboxes.squeeze()
    mask = prob > threshold
    valid_bbox = bboxes[mask]
    valid_prob = prob[mask]

    size = image.size
    valid_bbox = xcycwh_to_x1y1x2y2(valid_bbox, size)
    image = draw_bbox(image, valid_bbox, valid_prob)
    return image
