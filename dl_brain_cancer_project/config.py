%%writefile config.py
import torch

class CFG:
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    IMG_SIZE = 224
    NUM_CLASSES = 3
    ROOT_DIR = '/kaggle/input/datasets/orvile/brain-cancer-mri-dataset/Brain_Cancer raw MRI data/Brain_Cancer'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = 'ConvNeXt_Snall'
