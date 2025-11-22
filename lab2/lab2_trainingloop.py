#pip install evaluate
#pip install seqeval
from accelerate import Accelerator
from tqdm.auto import tqdm
import evaluate
import torch

accelerator = Accelerator()

def training_loop(model, train_dataloader, val_dataloader, lr_scheduler,  optimizer, label_names, num_train_epochs=3):
    metric = evaluate.load("seqeval")
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # Forward pass: compute loss
            outputs = model(**batch)
            loss = outputs.loss
            # Backward pass: compute gradients
            accelerator.backward(loss)
            
            # Update weights
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()  # Reset gradients for next step

            print(f'Loss: {loss}')
            progress_bar.update(1)
    
        model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        
        for batch in val_dataloader:
            with torch.no_grad():  # Don't compute gradients during evaluation
                outputs = model(**batch)
            
            # Get predicted labels (highest logit per token)
            
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            predictions = accelerator.pad_across_processes(
                predictions, dim=1, pad_index=-100
            )
            labels = accelerator.pad_across_processes(
                labels, dim=1, pad_index=-100
            )
            # Gather predictions from all processes
            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            
            # Convert to format expected by metric
            true_predictions, true_labels = postprocess(
                predictions_gathered, labels_gathered, label_names
            )
            # Add batch results to metric
            metric.add_batch(predictions=true_predictions, references=true_labels)
        
        # Compute final metrics for the epoch
        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )

def postprocess(predictions, labels, label_names):
    """
    Convert model predictions and labels to format expected by seqeval.
    - Remove special tokens (label = -100)
    - Convert integer labels to string labels (e.g., 3 -> "B-ORG")
    """
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    
    # Filter out -100 (special tokens) and convert to label names
    true_labels = [
        [label_names[l] for l in label if l != -100] 
        for label in labels
    ]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return true_labels, true_predictions