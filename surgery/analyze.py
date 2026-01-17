"""
Model Analysis Tools for Evlf Eris
Analyzes layer importance, attention head redundancy, and neuron activations
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
import argparse


class ModelAnalyzer:
    def __init__(self, model_path, config_path="configs/personality.yaml"):
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n{'='*50}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size: {total_params * 2 / 1e9:.2f} GB (float16)")
        print(f"{'='*50}\n")
        
        return total_params, trainable_params
    
    def analyze_layer_importance(self, test_texts=None):
        """
        Analyze importance of each transformer layer using gradient-based scoring
        """
        if test_texts is None:
            test_texts = [
                "Hello, how are you today?",
                "I love you so much",
                "Tell me about yourself",
                "What do you want to do today?",
                "You're so beautiful"
            ]
        
        print("Analyzing layer importance...")
        num_layers = len(self.model.model.layers)
        layer_scores = torch.zeros(num_layers)
        
        self.model.eval()
        
        for text in tqdm(test_texts, desc="Processing texts"):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Forward pass with gradient tracking
            outputs = self.model(**inputs, output_hidden_states=True)
            loss = outputs.logits[:, -1, :].sum()
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Calculate gradient magnitude for each layer
            for i, layer in enumerate(self.model.model.layers):
                grad_norm = 0
                for param in layer.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.abs().sum().item()
                layer_scores[i] += grad_norm
        
        # Normalize scores
        layer_scores = layer_scores / len(test_texts)
        layer_scores = layer_scores / layer_scores.max()
        
        print(f"\n{'='*50}")
        print("Layer Importance Scores (0-1):")
        print(f"{'='*50}")
        for i, score in enumerate(layer_scores):
            bar = '█' * int(score * 30)
            print(f"Layer {i:2d}: {score:.4f} {bar}")
        print(f"{'='*50}\n")
        
        # Identify low-importance layers
        threshold = 0.3
        low_importance = [i for i, s in enumerate(layer_scores) if s < threshold]
        print(f"Low importance layers (< {threshold}): {low_importance}")
        
        return layer_scores.numpy()
    
    def analyze_attention_heads(self, test_texts=None):
        """
        Analyze attention head redundancy and importance
        """
        if test_texts is None:
            test_texts = [
                "I love you",
                "You're amazing",
                "Tell me a story"
            ]
        
        print("\nAnalyzing attention heads...")
        num_layers = len(self.model.model.layers)
        num_heads = self.model.config.num_attention_heads
        
        head_importance = torch.zeros(num_layers, num_heads)
        
        self.model.eval()
        
        for text in tqdm(test_texts, desc="Processing texts"):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Forward with attention output
            outputs = self.model(**inputs, output_attentions=True)
            
            # Analyze attention patterns
            for layer_idx, attn_weights in enumerate(outputs.attentions):
                # attn_weights shape: [batch, num_heads, seq_len, seq_len]
                # Calculate entropy as importance measure
                for head_idx in range(num_heads):
                    attn = attn_weights[0, head_idx]
                    # Higher entropy = more distributed attention = more important
                    entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
                    head_importance[layer_idx, head_idx] += entropy.item()
        
        # Normalize
        head_importance = head_importance / len(test_texts)
        
        print(f"\n{'='*50}")
        print("Attention Head Importance (top 5 per layer):")
        print(f"{'='*50}")
        for layer_idx in range(num_layers):
            scores = head_importance[layer_idx]
            top_heads = torch.argsort(scores, descending=True)[:5]
            print(f"Layer {layer_idx:2d}: Heads {top_heads.tolist()} (scores: {scores[top_heads].tolist()})")
        print(f"{'='*50}\n")
        
        return head_importance.numpy()
    
    def analyze_neuron_activations(self, test_texts=None):
        """
        Analyze FFN neuron activation statistics
        """
        if test_texts is None:
            test_texts = [
                "I love you so much, you mean everything to me",
                "You're the best boyfriend ever",
                "Let's spend time together today"
            ]
        
        print("\nAnalyzing neuron activations...")
        num_layers = len(self.model.model.layers)
        
        neuron_stats = []
        
        self.model.eval()
        
        with torch.no_grad():
            for text in tqdm(test_texts, desc="Processing texts"):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                
                # Forward pass with hooks to capture activations
                activations = {}
                
                def hook_fn(name):
                    def hook(module, input, output):
                        activations[name] = output.detach()
                    return hook
                
                # Register hooks on MLP layers
                handles = []
                for i, layer in enumerate(self.model.model.layers):
                    handle = layer.mlp.register_forward_hook(hook_fn(f"layer_{i}"))
                    handles.append(handle)
                
                outputs = self.model(**inputs)
                
                # Remove hooks
                for handle in handles:
                    handle.remove()
                
                # Analyze activations
                for i in range(num_layers):
                    act = activations[f"layer_{i}"]
                    mean_act = act.abs().mean(dim=(0, 1))  # Average over batch and sequence
                    
                    if len(neuron_stats) <= i:
                        neuron_stats.append(mean_act.cpu())
                    else:
                        neuron_stats[i] += mean_act.cpu()
        
        # Average over all texts
        neuron_stats = [stats / len(test_texts) for stats in neuron_stats]
        
        print(f"\n{'='*50}")
        print("Neuron Activation Statistics:")
        print(f"{'='*50}")
        for i, stats in enumerate(neuron_stats):
            low_act = (stats < 0.01).sum().item()
            total = stats.numel()
            print(f"Layer {i:2d}: {low_act}/{total} ({100*low_act/total:.1f}%) neurons have low activation (<0.01)")
        print(f"{'='*50}\n")
        
        return neuron_stats
    
    def generate_pruning_recommendations(self, layer_scores, head_importance):
        """
        Generate pruning recommendations based on analysis
        """
        target_reduction = self.config['pruning']['target_reduction']
        
        print(f"\n{'='*60}")
        print(f"PRUNING RECOMMENDATIONS (Target: {target_reduction*100:.0f}% reduction)")
        print(f"{'='*60}\n")
        
        # Layer pruning recommendations
        threshold = np.percentile(layer_scores, target_reduction * 100)
        layers_to_prune = np.where(layer_scores < threshold)[0]
        
        # Exclude protected layers
        preserve = self.config['pruning']['preserve_layers']
        num_layers = len(layer_scores)
        preserve_indices = [i if i >= 0 else num_layers + i for i in preserve]
        layers_to_prune = [l for l in layers_to_prune if l not in preserve_indices]
        
        print(f"1. LAYER PRUNING:")
        print(f"   Suggested layers to remove: {layers_to_prune}")
        print(f"   Potential savings: {len(layers_to_prune)/len(layer_scores)*100:.1f}% of layers\n")
        
        # Attention head pruning
        min_head_importance = self.config['pruning']['attention_heads']['min_importance']
        heads_to_prune = []
        for layer_idx in range(len(head_importance)):
            for head_idx in range(len(head_importance[layer_idx])):
                if head_importance[layer_idx, head_idx] < min_head_importance:
                    heads_to_prune.append((layer_idx, head_idx))
        
        print(f"2. ATTENTION HEAD PRUNING:")
        print(f"   Heads below importance threshold ({min_head_importance}): {len(heads_to_prune)}")
        print(f"   Potential savings: {len(heads_to_prune)/(len(head_importance)*len(head_importance[0]))*100:.1f}% of heads\n")
        
        # Total estimated reduction
        layer_reduction = len(layers_to_prune) / len(layer_scores)
        head_reduction = len(heads_to_prune) / (len(head_importance) * len(head_importance[0])) * 0.2  # Heads are ~20% of params
        total_reduction = layer_reduction + head_reduction
        
        print(f"3. ESTIMATED TOTAL REDUCTION:")
        print(f"   {total_reduction*100:.1f}% of model parameters\n")
        print(f"{'='*60}\n")
        
        return {
            'layers_to_prune': layers_to_prune.tolist(),
            'heads_to_prune': heads_to_prune,
            'estimated_reduction': total_reduction
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze model for pruning")
    parser.add_argument("--model", type=str, default="models/base",
                        help="Path to model")
    parser.add_argument("--config", type=str, default="configs/personality.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    analyzer = ModelAnalyzer(args.model, args.config)
    
    # Run all analyses
    analyzer.count_parameters()
    layer_scores = analyzer.analyze_layer_importance()
    head_importance = analyzer.analyze_attention_heads()
    neuron_stats = analyzer.analyze_neuron_activations()
    
    # Generate recommendations
    recommendations = analyzer.generate_pruning_recommendations(layer_scores, head_importance)
    
    # Save results
    results_path = Path("analysis_results.yaml")
    with open(results_path, 'w') as f:
        yaml.dump({
            'layer_scores': layer_scores.tolist(),
            'head_importance': head_importance.tolist(),
            'recommendations': recommendations
        }, f)
    
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
