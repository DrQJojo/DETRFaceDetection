from accelerate import Accelerator

threshold = 0.85
lambda_bbox = 1
lambda_label = 1
lambda_iou = 1

num_epochs = 10
batch_size = 1
learning_rate = 1e-5
accelerator = Accelerator(gradient_accumulation_steps=2)
device = accelerator.device
PATH = r'model_checkpoint.pth'
