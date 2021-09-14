###Training Code###
import pixellib
from pixellib.custom_train import instance_custom_training

train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 2, batch_size = 4)
train_maskrcnn.load_pretrained_model("mask_rcnn_coco.h5")
train_maskrcnn.load_dataset("Flaggingimages")
train_maskrcnn.train_model(num_epochs = 80, augmentation=True,  path_trained_models = "mask_rcnn_models")
