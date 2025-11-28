import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import json
from datetime import datetime

class ModelComparison:
    "Framework for comparing my models on Charades"
    
    def __init__(self):
        self.results = {}

    def add_model_results(self, model_name, model_results):
        "Adds results from a model"
        results_dict = model_results[0] if isinstance(model_results, list) else model_results

        cleaned_results = {}
        for key, value in results_dict.items():
            clean_key = key.split('/')[-1]
            cleaned_results[clean_key] = value
            
        self.results[model_name] = cleaned_results

    def create_comparison_dataframe(self):
        data = []
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                # mAP scores (most important)
                'Overall mAP': results.get('test_overall_mAP', 0),
                'Verb mAP': results.get('test_verb_mAP', 0),
                'Object mAP': results.get('test_object_mAP', 0),
                'Action mAP': results.get('test_action_mAP', 0),
                # Macro F1 metrics
                'Verb F1 (macro)': results.get('test_verb_f1_macro', 0),
                'Object F1 (macro)': results.get('test_object_f1_macro', 0),
                'Action F1 (macro)': results.get('test_action_f1_macro', 0),
                # Micro F1 metrics
                'Verb F1 (micro)': results.get('test_verb_f1_micro', 0),
                'Object F1 (micro)': results.get('test_object_f1_micro', 0),
                'Action F1 (micro)': results.get('test_action_f1_micro', 0),
                # Precision/Recall
                'Verb Precision': results.get('test_verb_precision_macro', 0),
                'Object Precision': results.get('test_object_precision_macro', 0),
                'Action Precision': results.get('test_action_precision_macro', 0),
                'Verb Recall': results.get('test_verb_recall_macro', 0),
                'Object Recall': results.get('test_object_recall_macro', 0),
                'Action Recall': results.get('test_action_recall_macro', 0),
                # Other metrics
                'Verb Hamming Loss': results.get('test_verb_hamming_loss', 1),
                'Object Hamming Loss': results.get('test_object_hamming_loss', 1),
                'Action Hamming Loss': results.get('test_action_hamming_loss', 1),
                'Total Loss': results.get('test_total_loss', float('inf')),
            }
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('Overall mAP', ascending=False)
        return df

    def print_comparison_table(self):
        df = self.create_comparison_dataframe()

        print("\n" + "="*80)
        print("MODEL COMPARISON - SORTED BY OVERALL mAP")
        print("="*80)
        
        # Main mAP comparison
        print("\n mAP SCORES (Primary Charades Metric):")
        print("-"*80)
        map_cols = ['Model', 'Overall mAP', 'Verb mAP', 'Object mAP', 'Action mAP']
        print(df[map_cols].to_string(index=False))
        
        # F1 Score comparison
        print("\n F1 SCORES (Macro Average):")
        print("-"*80)
        f1_cols = ['Model', 'Verb F1 (macro)', 'Object F1 (macro)', 'Action F1 (macro)']
        print(df[f1_cols].to_string(index=False))

        print("\n F1 SCORES (Micro Average (weighted by class frequency):")
        print("-"*80)
        f1_micro_cols = ['Model', 'Verb F1 (micro)', 'Object F1 (micro)', 'Action F1 (micro)']
        print(df[f1_micro_cols].to_string(index=False))
        
        # Precision/Recall
        print("\n PRECISION & RECALL (Macro Average):")
        print("-"*80)
        pr_cols = ['Model', 'Verb Precision', 'Verb Recall', 'Object Precision', 
                   'Object Recall', 'Action Precision', 'Action Recall']
        print(df[pr_cols].to_string(index=False))
        
        # Best model summary
        best_model = df.iloc[0]
        print("\n" + "="*80)
        print(f"   BEST MODEL: {best_model['Model']}")
        print(f"   Overall mAP: {best_model['Overall mAP']:.4f}")
        print(f"   Verb mAP:    {best_model['Verb mAP']:.4f}")
        print(f"   Object mAP:  {best_model['Object mAP']:.4f}")
        print(f"   Action mAP:  {best_model['Action mAP']:.4f}")
        print("="*80)
        
        return df

    def plot_comparison(self, save_path='model_comparison.png'):
        """Create visualization comparing models"""
        df = self.create_comparison_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Comparison on Charades Dataset', fontsize=16, fontweight='bold')
        
        # 1. mAP comparison (most important)
        ax1 = axes[0, 0]
        map_data = df[['Model', 'Verb mAP', 'Object mAP', 'Action mAP', 'Overall mAP']].set_index('Model')
        map_data.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('mAP Scores (Primary Metric)', fontweight='bold')
        ax1.set_ylabel('mAP')
        ax1.set_xlabel('')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, max(1, map_data.max().max() * 1.1))
        
        # 2. F1 Score comparison
        ax2 = axes[0, 1]
        x = np.arange(len(df))
        width = 0.35
        df['Avg F1 (macro)'] = (df['Verb F1 (macro)'] + df['Object F1 (macro)'] + df['Action F1 (macro)']) / 3
        df['Avg F1 (micro)'] = (df['Verb F1 (micro)'] + df['Object F1 (micro)'] + df['Action F1 (micro)']) / 3
        
        ax2.bar(x - width/2, df['Avg F1 (macro)'], width, label='Macro Average', alpha=0.8, color='steelblue')
        ax2.bar(x + width/2, df['Avg F1 (micro)'], width, label='Micro Average', alpha=0.8, color='coral')
        
        ax2.set_title('F1 Scores: Macro vs Micro (Averages)', fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.set_xlabel('')
        ax2.set_xticks(x)
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Precision vs Recall
        ax3 = axes[1, 0]
        for i, row in df.iterrows():
            ax3.scatter(row['Verb Recall'], row['Verb Precision'], 
                       label=f"{row['Model']} (Verb)", s=100, alpha=0.6)
            ax3.scatter(row['Object Recall'], row['Object Precision'], 
                       label=f"{row['Model']} (Object)", s=100, alpha=0.6, marker='s')
            ax3.scatter(row['Action Recall'], row['Action Precision'], 
                       label=f"{row['Model']} (Action)", s=100, alpha=0.6, marker='^')
        ax3.set_title('Precision vs Recall Trade-off', fontweight='bold')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # 4. Hamming Loss comparison (lower is better)
        ax4 = axes[1, 1]
        hamming_data = df[['Model', 'Verb Hamming Loss', 'Object Hamming Loss', 
                           'Action Hamming Loss']].set_index('Model')
        hamming_data.plot(kind='bar', ax=ax4, width=0.8, color=['#ff9999', '#99ccff', '#99ff99'])
        ax4.set_title('Hamming Loss (Lower is Better)', fontweight='bold')
        ax4.set_ylabel('Hamming Loss')
        ax4.set_xlabel('')
        ax4.legend(loc='upper right')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n Comparison plot saved to: {save_path}")
        plt.show()

    def save_results(self, filename='model_comparison_results.json'):
        """Save all results to JSON"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            'timestamp': timestamp,
            'models': self.results,
            'summary': {
                'num_models': len(self.results),
                'best_model': max(self.results.items(), 
                                 key=lambda x: x[1].get('test_overall_mAP', 0))[0]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n Results saved to: {filename}")
    
    def export_to_csv(self, filename='model_comparison.csv'):
        """Export comparison to CSV"""
        df = self.create_comparison_dataframe()
        df.to_csv(filename, index=False)
        print(f"\n CSV exported to: {filename}")
        