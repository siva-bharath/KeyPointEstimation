import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import mlflow
import matplotlib.pyplot as plt

from setup.config import Config
from model.posenet import LightweightPoseNet
from train.trainer import train_epoch, evaluate
from train.loss import KeypointFocalLoss, KeypointMSELoss
from train.tuner import EarlyStopping
from dataset.download_dataset import download_coco_dataset
from dataset.keypt_dataloader import COCOKeypointDataset

def mlflow_init(tracking_uri: str, experiment_name: str):
    """Initialize MLflow"""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def export_to_onnx(model, save_path, input_shape=(1, 3, 256, 256)):
    """Export PyTorch model to ONNX format"""
    model.eval()
    
    # Get the device the model is on
    device = next(model.parameters()).device
    
    # Create dummy input on the same device as the model
    dummy_input = torch.randn(input_shape, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to ONNX: {save_path}")


def main():
    tb_writer = SummaryWriter() 
    cfg = Config()
    cfg.num_epochs = 5 # Config the number of epochs
    
    if not os.path.exists(cfg.data_dir):
        download_coco_dataset(cfg.data_dir)
    
    # MLflow setup 
    tracking_uri = "http://127.0.0.1:8080"  
    experiment_name = "posenet"
    mlflow_init(tracking_uri, experiment_name)

    with mlflow.start_run(run_name="lightweight_posenet"):
        
        # Log configuration parameters
        mlflow.log_params({
            "num_epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "img_size": cfg.img_size,
            "num_keypoints": cfg.num_keypoints,
            "device": cfg.device,
            "early_stopping_patience": cfg.early_stopping_patience,
        })

    
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = COCOKeypointDataset(cfg, transform=transform, is_train=True)

        val_dataset = COCOKeypointDataset(cfg,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            is_train=False
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

        # Create model
        model = LightweightPoseNet(num_keypoints=cfg.num_keypoints)
        model = model.to(cfg.device)

        # Loss and optimizer
        if cfg.loss_type == 'mse':
            criterion = KeypointMSELoss()
        else:  
            criterion = KeypointFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
            
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=10,  # Increased patience for better convergence
            factor=0.7,   # Less aggressive reduction
            min_lr=1e-6   # Set minimum LR to prevent getting stuck
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)

        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)

        print(f"Training on {cfg.device}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Training loop
        for epoch in range(cfg.num_epochs):
            print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")

            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg.device)

            # Evaluate
            validation_metrics = evaluate(model, val_loader, criterion, cfg.device)

            # Logging through tensor board
            tb_writer.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
            tb_writer.add_scalar(tag='Loss/valid', scalar_value=validation_metrics['loss'], global_step=epoch)
            tb_writer.add_scalar("kp/AP@0.50", scalar_value=validation_metrics['AP@0.50'], global_step=epoch)
            tb_writer.add_scalar('Precision', scalar_value=validation_metrics['precision'], global_step=epoch)
            tb_writer.add_scalar("Recall", scalar_value=validation_metrics['recall'], global_step=epoch)
            
            # logging through mlflow
            mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": validation_metrics['loss'],
                    "val_AP_0_50": validation_metrics['AP@0.50'],
                    "val_precision": validation_metrics['precision'],
                    "val_recall": validation_metrics['recall'],
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=epoch)
            
            # Update learning rate
            scheduler.step(validation_metrics['loss'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {validation_metrics['loss']:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save model every n epochs
            if (epoch + 1) % cfg.save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': validation_metrics['loss']
                }
                torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')
                print(f"Model saved at epoch {epoch+1}")

            # Early stopping check
            if early_stopping(validation_metrics['loss']):
                print("Early stopping triggered!")
                break

            # Save best model
            if validation_metrics['loss'] < early_stopping.best_loss:
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')
                print("Best model saved!")

        # Export to ONNX after training
        print("\nExporting model to ONNX...")
        onnx_path = 'checkpoints/posenet_model.onnx'
        try:
            export_to_onnx(model, onnx_path, input_shape=(1, 3, cfg.img_size, cfg.img_size))
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")
            
        # Log the best PyTorch model (only if it exists)
        if os.path.exists('checkpoints/best_model.pth'):
            mlflow.log_artifact('checkpoints/best_model.pth', "pytorch_model")
            
        # Log the final model state with input example
        device = next(model.parameters()).device
        # Create input example as numpy array (also supported by MLflow)
        input_example = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)
        input_example_np = input_example.cpu().numpy()
        mlflow.pytorch.log_model(model, "model", input_example=input_example_np)

        print(f"\nTraining completed!")

    tb_writer.close()

if __name__ == '__main__' :
    main()

