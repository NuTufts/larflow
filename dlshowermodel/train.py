import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        edge_pred = model(batch)
        
        # Compute loss
        loss = criterion(edge_pred, batch.edge_label)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add this line
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
        # Store predictions and labels for metrics
        preds = torch.sigmoid(edge_pred).detach().cpu().numpy()
        labels = batch.edge_label.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # Calculate metrics
    preds_binary = (np.array(all_preds) >= 0.5).astype(float)
    accuracy = accuracy_score(all_labels, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds_binary, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5  # Default if there's only one class
    
    avg_loss = total_loss / len(loader.dataset)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


# Evaluation function
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        
        # Forward pass
        edge_pred = model(batch)
        
        # Compute loss
        loss = criterion(edge_pred, batch.edge_label)
        
        total_loss += loss.item() * batch.num_graphs
        
        # Store predictions and labels for metrics
        preds = torch.sigmoid(edge_pred).detach().cpu().numpy()
        labels = batch.edge_label.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # Calculate metrics
    preds_binary = (np.array(all_preds) >= 0.5).astype(float)
    accuracy = accuracy_score(all_labels, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds_binary, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5  # Default if there's only one class
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, preds_binary)
    
    avg_loss = total_loss / len(loader.dataset)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }

# Function to plot training curves
def plot_training_curves(train_metrics, val_metrics):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_metrics) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='Training')
    plt.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['accuracy'] for m in train_metrics], 'b-', label='Training')
    plt.plot(epochs, [m['accuracy'] for m in val_metrics], 'r-', label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['f1'] for m in train_metrics], 'b-', label='Training')
    plt.plot(epochs, [m['f1'] for m in val_metrics], 'r-', label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot AUC-ROC
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['auc'] for m in train_metrics], 'b-', label='Training')
    plt.plot(epochs, [m['auc'] for m in val_metrics], 'r-', label='Validation')
    plt.title('AUC-ROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC-ROC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

# Main training function
def train_model(model, train_loader, val_loader, test_loader, 
                criterion, optimizer, device, 
                num_epochs=1000, patience=10,
                lr=1.0e-3, 
                burn_in_epochs=100,
                burn_in_lr=1.0e-6,
                logger=None,
                model_save_path='best_model.pt'):
    
    best_val_f1 = 0
    counter = 0
    
    # Lists to store metrics
    train_metrics = []
    val_metrics = []

    if logger is not None:
        logger.watch(model, log="all", log_freq=100)
    
    for epoch in range(burn_in_epochs+num_epochs):

        if epoch==burn_in_epochs:
            print(f"END OF BURN-IN. Switch lr from {burn_in_lr} to {lr}")
            for g in optimizer.param_groups:
                g['lr'] = lr

        # Train
        train_metric = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics.append(train_metric)
        
        # Validate
        val_metric = evaluate(model, val_loader, criterion, device)
        val_metrics.append(val_metric)
        
        # Print progress
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_metric['loss']:.4f}, Acc: {train_metric['accuracy']:.4f}, F1: {train_metric['f1']:.4f}")
        print(f"  Val   - Loss: {val_metric['loss']:.4f}, Acc: {val_metric['accuracy']:.4f}, F1: {val_metric['f1']:.4f}")

        log_info = ['loss',
            'accuracy',
            'precision',
            'recall',
            'f1']
        
        logger_data = {"epoch":epoch}
        for info in log_info:
            logger_data[f'train/{info}'] = train_metric[info]
            logger_data[f'val/{info}'] = val_metric[info]
        logger.log(logger_data,step=epoch)
        
        
        # Early stopping based on validation F1 score
        if val_metric['f1'] > best_val_f1 or epoch==0:
            best_val_f1 = val_metric['f1']
            counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved best model with F1: {best_val_f1:.4f}")
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("\nTest set metrics:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    # Plot training curves
    plot_training_curves(train_metrics, val_metrics)
    
    return model, train_metrics, val_metrics, test_metrics
