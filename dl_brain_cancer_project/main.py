%%writefile main.py
import torch
import torch.nn as nn
from config import CFG
from model import My_model
from dataset import ImageDataset, prepare_dataloaders
from engine import Trainer
from utils import plot_history

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
