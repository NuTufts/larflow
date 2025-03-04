import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm

from dlshowermodel.utils.AverageMeter import AverageMeter

training_metrics = ['loss',
                    'true_acc',
                    'false_acc',
                    'precision',
                    'f1',
                    'ntrue_frac']

@torch.no_grad()
def evaluate_batch( edge_logit, edge_label ):
    metrics = {}
    for metric in training_metrics:
        metrics[metric] = None

    edge_prob = torch.sigmoid(edge_logit)
    edge_pred = edge_prob>0.5
    #print('edge_prob.shape=',edge_prob.shape)
    #print('edge_label.shape=',edge_label.shape)

    npos_mask = (edge_label==1.0)
    nneg_mask = (edge_label==0.0)
    npos_sum = npos_mask.sum().cpu().item()
    nneg_sum = nneg_mask.sum().cpu().item()

    pred_pos = edge_pred[npos_mask]
    pred_neg = edge_pred[nneg_mask]

    if npos_sum>0:
        metrics['true_acc'] = float(pred_pos.sum().cpu().item())/float(npos_sum) # true-positives

    if nneg_sum>0:
        metrics['false_acc'] = float((pred_neg==False).sum().cpu().item())/float(nneg_sum) # true-negatives

    npos_pred = (edge_pred==True).sum().cpu().item()
    if npos_pred>0:
        tp = pred_pos.sum().cpu().item()
        metrics['precision'] = float(tp)/float(npos_pred)
    else:
        tp = 0.0

    if edge_pred.shape[0]>0:
        metrics['ntrue_frac'] = npos_sum/edge_pred.shape[0]

    fp = float( (pred_neg==True).sum().cpu().item() ) # false-positives: number of true 'neg' edges (pred_neg) that were labeled true
    fn = float( (pred_pos==False).sum().cpu().item()) # false-negatives: number of true 'pos' edges (pred_pos) that were labeled false
    f1_denom = 2*tp + fp + fn
    if f1_denom>0.0:
        f1 = 2*tp/f1_denom
        metrics['f1'] = f1
        
    return metrics

@torch.no_grad()
def evaluate_valid( model, valid_iter, valid_loader, criterion, device,
                    nvalid_batches=10 ):
    """
    Calculate evaluation metrics for monitoring performance of model on validation dataset.
    """
    meters = AverageMeter.make_meter_dict( training_metrics )
    model.eval()
    for ibatch in range(nvalid_batches):
        try:
            batch = next(valid_iter).to(device)
        except:
            print("Valid dataset iterator expired. Reset")
            valid_iter = iter(valid_loader)
            batch = next(valid_iter).to(device)

        edge_logit = model(batch)
        loss = criterion(edge_logit,batch.edge_label)
        meters['loss'].update(loss.detach().cpu().item())

        metrics = evaluate_batch( edge_logit.detach(), batch.edge_label.detach() )
        for metric,val in metrics.items():
            if val is not None:
                meters[metric].update(val)

    return meters
            

# Training function
def train_epoch(training_config, model, lr, train_loader, valid_loader, valid_iter,
                optimizer, criterion, device, wandb_logger,
                current_iter_num, lr_scheduler=None ):
    model.train()

    niter_per_eval = training_config['niters_per_eval']
    train_meters = AverageMeter.make_meter_dict(training_metrics)
    last_valid_meters = None
    niter_per_epoch = training_config['niters_per_train_epoch']

    if lr_scheduler is None:
        for g in optimizer.param_groups:
            g['lr'] = lr

    iiter = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr = lr_scheduler.get_lr(current_iter_num+iiter)
            # add layer modifiers here
            for g in optimizer.param_groups:
                g['lr'] = lr
        
        # Forward pass
        edge_pred = model(batch)
        
        # Compute loss
        loss = criterion(edge_pred, batch.edge_label)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add this line
        optimizer.step()

        # eval metrics for training
        with torch.no_grad():
        
            # Store predictions and labels for metrics
            metrics = evaluate_batch( edge_pred.detach(), batch.edge_label.detach() )
            for metric,val in metrics.items():
                if val is not None:
                    train_meters[metric].update(val)
            train_meters['loss'].update( loss.detach().cpu().item() )
                    
            # evaluate metrics on validation set
            # at regular interval or when we've reached the end of the epoch
            if iiter>0 and (iiter%niter_per_eval==0 or (iiter+1)>=niter_per_epoch):
                # validation evaluation
                model.eval()
                valid_meters = evaluate_valid( model, valid_iter, valid_loader, criterion, device,
                                             nvalid_batches=training_config['eval_nvalid_batches'] )

                # log training and valid metrics to wandb logger
                if training_config['log_to_wandb']:                
                    logdata = {'epoch':float(current_iter_num+iiter)/float(niter_per_epoch),
                                'lr':lr}
                    for sample,meters in [('train',train_meters),('valid',valid_meters)]:
                        for metric,meter in meters.items():
                            logmetric_name = f'{sample}/{metric}'
                            logdata[logmetric_name] = None
                            if meter.count>0:
                                logdata[logmetric_name] = meter.avg
                    wandb_logger.log(logdata,step=current_iter_num+iiter)
                # reset training meters
                for metric,meters in train_meters.items():
                    meters.reset()
                # set model back to train mode
                model.train()
                last_valid_meters = valid_meters

        # end of eval block
        iiter += 1

    
    # # Calculate Epoch Metrics
    # preds_binary = (np.array(all_preds) >= 0.5).astype(float)
    # accuracy = accuracy_score(all_labels, preds_binary)
    # precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds_binary, average='binary', zero_division=0)
    
    # try:
    #     auc = roc_auc_score(all_labels, all_preds)
    # except:
    #     auc = 0.5  # Default if there's only one class
    
    # avg_loss = total_loss / len(loader.dataset)
    
    # return {
    #     'loss': avg_loss,
    #     'accuracy': accuracy,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1,
    #     'auc': auc
    # }
    return current_iter_num+iiter, last_valid_meters


# Evaluation function: unused for now
@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
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
def train_model(train_config, model, train_loader, valid_loader, valid_iter,
                criterion, optimizer, device, batch_size,
                num_epochs=1000, patience=10,
                lr=1.0e-3, 
                burn_in_epochs=100,
                burn_in_lr=1.0e-6,
                logger=None,
                lr_scheduler=None,
                model_save_path='best_model.pt'):
    
    best_val_f1 = 0
    counter = 0
    
    # Lists to store metrics
    train_metrics = []
    val_metrics = []

    if logger is not None:
        logger.watch(model, log="all", log_freq=100)

    current_niters = 0
    if 'starting_iter_num' in train_config:
        current_niters = train_config['starting_iter_num']

        
    for epoch in range(num_epochs):

        # Train
        current_niters, last_valid_meters = train_epoch(train_config, model, lr,
                                                        train_loader, valid_loader, valid_iter,
                                                        optimizer, criterion, device, logger, current_niters,
                                                        lr_scheduler=lr_scheduler )
        
        # Print progress
        print(f"Epoch: {epoch+1}/{num_epochs}. Currrent Niters: {current_niters}.")

        # periodic checkpoint
        ntraining_epochs = epoch-burn_in_epochs
        if ntraining_epochs>0 and ntraining_epochs%train_config['epochs_per_checkpoint']==0:
            model_save_path = f'ubshower_gnn_checkpoint_epoch{ntraining_epochs}_iter{current_niters}.pt'
            torch.save(model.state_dict(), model_save_path)
        
        # Early stopping based on validation F1 score
        if last_valid_meters is not None:
            if (last_valid_meters['f1'].count>0 and last_valid_meters['f1'].avg > best_val_f1) or epoch==0:
                best_val_f1 = last_valid_meters['f1'].avg
                counter = 0
                # Save best model
                model_save_path = 'ubshower_gnn_bestmodel_f1.pt'
                torch.save(model.state_dict(), model_save_path)
                print(f"  Saved best model with F1: {best_val_f1:.4f}")
            for metric,meter in last_valid_meters.items():
                if meter.count>0:
                    print(f'  {metric}: {meter.avg:.4f} (from {meter.count} counts)')
                else:
                    print(f'  {metric}: no qualifying batches')
        else:
           counter += 1
            
        if train_config['use_early_stopping'] and counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # # Load best model for final evaluation
    # model.load_state_dict(torch.load(model_save_path))
    
    # # Evaluate on test set
    # test_metrics = evaluate(model, test_loader, criterion, device)
    
    # print("\nTest set metrics:")
    # print(f"Loss: {test_metrics['loss']:.4f}")
    # print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    # print(f"Precision: {test_metrics['precision']:.4f}")
    # print(f"Recall: {test_metrics['recall']:.4f}")
    # print(f"F1 Score: {test_metrics['f1']:.4f}")
    # print(f"AUC-ROC: {test_metrics['auc']:.4f}")
    # print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    # # Plot training curves
    # plot_training_curves(train_metrics, val_metrics)
    
    return model
