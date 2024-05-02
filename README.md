# DETRFaceDetection
Fine tuning DETR model for face detection task

For the UsingTrainer.ipynb file, I just followed the steps provided by hugging face https://huggingface.co/docs/transformers/en/tasks/object_detection to finish the fine-tuning and evaluation.
The model is available at on hugging face NekoJojo/DETRFaceDetection.

For the code part, I:
  change the class output head from Linear(256,92) to Linear(256,1) for my face detection task, so now the class is 1: face, 0: no face.
  implement the loss function, that using Hungarian Match to find the bipartie loss, combining class loss, bbox iou loss and bbox L1 loss. Instead of giou loss that is described in the DETR paper, i use diou.
  try to fine-tuned the modified model.
Run the main.py to train this model.
I hope everything is correct, however due to the computation resources, I cannot finish the training loop unfortunately.
For the part, the dataset is preprocessed so that each image is padded with 100 labels and bboxes to fit the DETR model (while using trainer hugging face will take care of almost everything).

Results:
before fine tuning, the model captures human, but not human face.
![image](https://github.com/DrQJojo/DETRFaceDetection/assets/140708983/c573b57e-b4d3-43a6-8d08-caf6d1b9f259)
after fine tuning, the model captures human face
![image](https://github.com/DrQJojo/DETRFaceDetection/assets/140708983/5f26e30d-c854-40b8-add0-880f28ce2091)

