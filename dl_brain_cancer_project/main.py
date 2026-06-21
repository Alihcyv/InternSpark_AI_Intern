%%writefile main.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import CFG
from model import My_model
from dataset import ImageDataset, prepare_dataloaders
from engine import Trainer

def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    full_dataset = ImageDataset(CFG)
    train_loader, val_loader, test_loader = prepare_dataloaders(full_dataset, CFG)
    
    model = My_model(CFG).to(CFG.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    
    trainer = Trainer(
        model=model, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        cfg=CFG
    )
    
    print("Starting training...")
    best_val_acc = trainer.fit(train_loader, val_loader)
    
    print("\nPlotting learning curves...")
    plot_history(trainer.history)
    
    print("\nRunning final evaluation on test set...")
    model.load_state_dict(torch.load(CFG.MODEL_NAME))
    
    test_loss, test_acc = trainer.evaluate(test_loader)
    print("-" * 30)
    print(f"Final Results:")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Loss:    {test_loss:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
