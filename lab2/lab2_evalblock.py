#pip install evaluate
import torch
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy import stats
from collections import defaultdict
import json
import os
from typing import Dict, List, Tuple, Optional

class TokenClassificationEvaluator:
    """
    Comprehensive evaluator for token classification models (NER, Chunking, etc.)
    
    Usage:
        evaluator = TokenClassificationEvaluator(device='cuda')
        
        # Evaluate single model
        results = evaluator.evaluate(model, test_dataloader, label_names, model_name="My Model")
        evaluator.print_results(results)
        
        # Compare multiple models
        evaluator.add_model(english_model, english_test_dataloader, english_label_names, "English NER")
        evaluator.add_model(hindi_model, hindi_test_dataloader, hindi_label_names, "Hindi Chunking")
        comparison_df = evaluator.compare_models()
        evaluator.create_visualizations()
        evaluator.statistical_comparison()
        evaluator.save_results('evaluation_results.json')
    """
    def __init__(self, device: str = 'cuda:2'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results_dict = {}
        self.seqeval_metric = evaluate.load("seqeval")

        print(f"TokenClassificationEvaluator Initialized on {self.device}")

    #====================================
    # MAIN EVALUATION BLOCK
    #====================================
    
    def evaluate(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 label_names: List[str],
                 model_name: str = "Model") -> Dict:
        """
        Evaluate a single model comprehensively

        Args:
            model: PyTorch model to evaluate
            dataloader: Test dataloader
            label_names: List of label names
            model_name: Name of this model
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"\n{'='*60}")

        model.eval()
        model = model.to(self.device)

        all_predictions = []
        all_labels = []
        token_predictions = []
        token_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch['labels']

                predictions = predictions.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                for pred, label in zip(predictions, labels):
                    valid_indicies = label != -100
                    valid_preds = pred[valid_indicies]
                    valid_labels = label[valid_indicies]

                    pred_tags = [label_names[p] for p in valid_preds]
                    true_tags = [label_names[l] for l in valid_labels]

                    all_predictions.append(pred_tags)
                    all_labels.append(true_tags)

                    token_predictions.extend(valid_preds)
                    token_labels.extend(valid_labels)

                if (batch_idx +1) % 50 == 0:
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

        print("Computing metrics...")

        results = self._compute_metrics(all_predictions,
                                        all_labels,
                                        token_predictions,
                                        token_labels,
                                        label_names,
                                        model_name
                                       )
        return results

    def _compute_metrics(self,
                         all_predictions: List[List[str]],
                         all_labels: List[List[str]],
                         token_predictions: List[int],
                         token_labels: List[int],
                         label_names: List[str],
                         model_name: str) -> Dict:
        "Compute all evaluation metrics"
        #1. seqeval metrics
        seqeval_results = self.seqeval_metric.compute(predictions=all_predictions, references=all_labels)

        #2. token-level metrics
        token_predictions_np = np.array(token_predictions)
        token_labels_np = np.array(token_labels)
        token_accuracy = (token_predictions_np == token_labels_np).mean()

        #per class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(token_labels_np,
                                                                                                                 token_predictions_np,
                                                                                                                 labels=range(len(label_names)),
                                                                                                                 average=None,
                                                                                                                 zero_division=0
                                                                                                                 )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(token_labels_np,
                                                                                     token_predictions_np,
                                                                                     average='macro',
                                                                                     zero_division=0
                                                                                    )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(token_labels_np,
                                                                                              token_predictions_np,
                                                                                              average='weighted',
                                                                                              zero_division=0
                                                                                             )
        #3. confusion matrix
        cm = confusion_matrix(token_labels_np, token_predictions_np)

        #organise results
        results = {'model_name': model_name,
                   'overall_metrics': {'precision': seqeval_results['overall_precision'],
                                       'recall': seqeval_results['overall_recall'],
                                       'f1': seqeval_results['overall_f1'],
                                       'accuracy': seqeval_results['overall_accuracy']
                                      },
                   'token_level_metrics': {'accuracy': token_accuracy,
                                           'precision_macro': precision_macro,
                                           'recall_macro': recall_macro,
                                           'f1_macro': f1_macro,
                                           'precision_weighted': precision_weighted,
                                           'recall_weighted': recall_weighted,
                                           'f1_weighted': f1_weighted
                                          },
                   'per_class_metrics': {label_names[i]: {'precision':float(precision_per_class[i]),
                                                          'recall': float(recall_per_class[i]),
                                                          'f1': float(f1_per_class[i]),
                                                          'support': int(support_per_class[i])
                                                         }
                                                         for i in range(len(label_names)) if support_per_class[i] > 0
                                        },
                   'confusion_matrix': cm,
                   'predictions': all_predictions,
                   'labels': all_labels,
                   'label_names': label_names
                  }
        return results

    #====================================
    # MANAGE MULTIPLE MODELS
    #====================================
    def add_model(self,
                  model: torch.nn.Module,
                  dataloader: DataLoader,
                  label_names: List[str],
                  model_name: str):
        """
        Evaluate and add a model to the comparison.
        
        Args:
            model: PyTorch model
            dataloader: Test dataloader
            label_names: List of label names
            model_name: Name for this model
        """
        results = self.evaluate(model, dataloader, label_names, model_name)
        self.results_dict[model_name] = results
        print(f"Added {model_name} to comparison")

    def clear_models(self):
        "Clear all stored model results"
        self.results_dict = {}
        print("Cleared all model results")

    #====================================
    #  DISPLAY RESULTS
    #====================================
    def print_results(self, results: Dict):
        "Print detailed results for a single model"
        model_name = results['model_name']
        overall = results['overall_metrics']
        token = results['token_level_metrics']

        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY: {model_name}")
        print(f"\n{'='*60}")

        print("\n ENTITY-LEVEL METRICS (Seqeval):")
        print(f"  Precision:  {overall['precision']:.4f}")
        print(f"  Recall:     {overall['recall']:.4f}")
        print(f"  F1-Score:   {overall['f1']:.4f}")
        print(f"  Accuracy:   {overall['accuracy']:.4f}")
        
        print("\n TOKEN-LEVEL METRICS:")
        print(f"  Accuracy:             {token['accuracy']:.4f}")
        print(f"  Precision (macro):    {token['precision_macro']:.4f}")
        print(f"  Recall (macro):       {token['recall_macro']:.4f}")
        print(f"  F1-Score (macro):     {token['f1_macro']:.4f}")
        print(f"  Precision (weighted): {token['precision_weighted']:.4f}")
        print(f"  Recall (weighted):    {token['recall_weighted']:.4f}")
        print(f"  F1-Score (weighted):  {token['f1_weighted']:.4f}")
        
        self._print_per_class_metrics(results)

    def _print_per_class_metrics(self, results: Dict, top_n: int=10):
        "Print per class performance metrics"
        print(f"\n PER-CLASS PERFORMANCE (Top {top_n} by support):")
        print("-" * 70)
        
        per_class = results['per_class_metrics']
        
        # Sort by support
        sorted_classes = sorted(
            per_class.items(),
            key=lambda x: x[1]['support'],
            reverse=True
        )[:top_n]
        
        print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 70)
        
        for class_name, metrics in sorted_classes:
            print(f"{class_name:<20} "
                  f"{metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} "
                  f"{metrics['f1']:>10.4f} "
                  f"{metrics['support']:>10}")

    #===============================================
    # COMPARISON BLOCK
    #===============================================
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all added models
        
        Returns:
            DataFrame with comparison metrics
        """

        if len(self.results_dict) == 0:
            print("No models to compare. Use add_model() first.")
            return None
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")

        comparison_data = []

        for model_name, results in self.results_dict.items():
            overall = results['overall_metrics']
            token = results['token_level_metrics']

            comparison_data.append({
                'Model': model_name,
                'Entity F1': overall['f1'],
                'Entity Precision': overall['precision'],
                'Entity Recall': overall['recall'],
                'Entity Accuracy': overall['accuracy'],
                'Token Accuracy': token['accuracy'],
                'Token F1 (macro)': token['f1_macro'],
                'Token F1 (weighted)': token['f1_weighted']
            })
        df = pd.DataFrame(comparison_data)
        df = df.set_index('Model')

        print("\n OVERALL METRICS COMPARISON:")
        print(df.to_string())
        
        # Find best model for each metric
        if len(self.results_dict) > 1:
            print("\n BEST PERFORMING MODEL PER METRIC:")
            for col in df.columns:
                best_model = df[col].idxmax()
                best_score = df[col].max()
                print(f"  {col:<25}: {best_model:<20} ({best_score:.4f})")
        
        return df

    #===========================================
    # VISUALISATION BLOCK
    #===========================================
    def create_visualisation(self, save_dir: str='./evaluation_plots'):
        """
        Create all comparison visualizations.
        
        Args:
            save_dir: Directory to save plots
        """
        if len(self.results_dict) == 0:
            print("No models to visualize. Use add_model() first.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Creating Visualizations")
        print(f"{'='*60}")
        
        self._plot_overall_metrics(save_dir)
        self._plot_confusion_matrices(save_dir)
        self._plot_per_class_comparison(save_dir)
        
        print(f"\n All plots saved to {save_dir}")
            
    def _plot_overall_metrics(self, save_dir: str):
        """Plot overall metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison: Overall Metrics', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
        metric_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            model_names = list(self.results_dict.keys())
            scores = [self.results_dict[m]['overall_metrics'][metric] for m in model_names]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            bars = ax.bar(model_names, scores, color=colors)
            ax.set_ylabel(label, fontsize=12)
            ax.set_ylim([0, 1])
            ax.set_title(f'Entity-Level {label}', fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/overall_metrics_comparison.png")

    def _plot_overall_metrics(self, save_dir: str):
        """Plot overall metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison: Overall Metrics', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy']
        metric_labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        
        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            model_names = list(self.results_dict.keys())
            scores = [self.results_dict[m]['overall_metrics'][metric] for m in model_names]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            bars = ax.bar(model_names, scores, color=colors)
            ax.set_ylabel(label, fontsize=12)
            ax.set_ylim([0, 1])
            ax.set_title(f'Entity-Level {label}', fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: overall_metrics_comparison.png")
    
    def _plot_confusion_matrices(self, save_dir: str):
        """Plot confusion matrices for all models"""
        n_models = len(self.results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        
        for idx, (model_name, results) in enumerate(self.results_dict.items()):
            cm = results['confusion_matrix']
            label_names = results['label_names']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            cm_normalized = np.nan_to_num(cm_normalized)
            
            # Plot only classes that appear in test set
            active_classes = np.where(cm.sum(axis=1) > 0)[0]
            cm_subset = cm_normalized[active_classes][:, active_classes]
            active_labels = [label_names[i] for i in active_classes]
            
            # Limit display if too many classes
            max_classes = 20
            if len(active_labels) > max_classes:
                cm_subset = cm_subset[:max_classes, :max_classes]
                active_labels = active_labels[:max_classes]
            
            sns.heatmap(cm_subset, 
                       xticklabels=active_labels,
                       yticklabels=active_labels,
                       annot=False,
                       fmt='.2f',
                       cmap='Blues',
                       ax=axes[idx],
                       cbar_kws={'label': 'Normalized Frequency'})
            
            axes[idx].set_title(f'{model_name}', fontweight='bold')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(axes[idx].get_yticklabels(), rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: confusion_matrices.png")
    
    def _plot_per_class_comparison(self, save_dir: str, top_n: int = 15):
        """Plot per-class F1 score comparison"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get common classes across all models
        all_classes = set()
        for results in self.results_dict.values():
            all_classes.update(results['per_class_metrics'].keys())
        
        # Select top classes by average support
        class_support = defaultdict(list)
        for results in self.results_dict.values():
            for cls, metrics in results['per_class_metrics'].items():
                class_support[cls].append(metrics['support'])
        
        avg_support = {cls: np.mean(supports) for cls, supports in class_support.items()}
        top_classes = sorted(avg_support.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_class_names = [cls for cls, _ in top_classes]
        
        # Prepare data for plotting
        x = np.arange(len(top_class_names))
        width = 0.8 / len(self.results_dict)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.results_dict)))
        
        for idx, (model_name, results) in enumerate(self.results_dict.items()):
            f1_scores = []
            for cls in top_class_names:
                if cls in results['per_class_metrics']:
                    f1_scores.append(results['per_class_metrics'][cls]['f1'])
                else:
                    f1_scores.append(0)
            
            offset = width * (idx - len(self.results_dict)/2 + 0.5)
            ax.bar(x + offset, f1_scores, width, label=model_name, alpha=0.8, color=colors[idx])
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Per-Class F1-Score Comparison (Top {top_n} Classes)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: per_class_f1_comparison.png")

    #====================================================================
    # SAVE RESULTS
    #====================================================================
    def save_results(self, filepath: str = 'evaluation_results.json'):
        """Save evaluation results to JSON file"""
        results_to_save = {}
        
        for model_name, result in self.results_dict.items():
            results_to_save[model_name] = {
                'overall_metrics': result['overall_metrics'],
                'token_level_metrics': result['token_level_metrics'],
                'per_class_metrics': result['per_class_metrics'],
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n Results saved to {filepath}")