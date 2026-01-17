"""
Structured Pruning for Evlf Eris
Removes entire layers and attention heads to reduce model size
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import copy


class StructuredPruner:
    def __init__(self, model_path, config_path="configs/personality.yaml"):
        print(f"Loading model for pruning from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        
        with open(config_path, 'r') as f:
            self.prune_config = yaml.safe_load(f)
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self device}")
    
    def prune_layers(self, layers_to_remove):
        """
        Remove entire transformer layers
        """
        print(f"\nPruning {len(layers_to_remove)} layers: {layers_to_remove}")
        
        total_layers = len(self.model.model.layers)
        
        # Create mask for layers to keep
        keep_mask = [i not in layers_to_remove for i in range(total_layers)]
        
        # Filter layers
        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.model.model.layers) if keep_mask[i]
        ])
        
        # Replace layer list
        self.model.model.layers = new_layers
        
        # Update config
        self.config.num_hidden_layers = len(new_layers)
        
        print(f"Layers reduced from {total_layers} to {len(new_layers)}")
        
        return len(new_layers)
    
    def prune_attention_heads(self, heads_to_prune):
        """
        Prune specific attention heads by zeroing out their parameters
        """
        if not heads_to_prune:
            print("No attention heads to prune")
            return
        
        print(f"\nPruning {len(heads_to_prune)} attention heads...")
        
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads
        
        for layer_idx, head_idx in tqdm(heads_to_prune, desc="Pruning heads"):
            if layer_idx >= len(self.model.model.layers):
                continue
            
            layer = self.model.model.layers[layer_idx]
            attn = layer.self_attn
            
            # Zero out Q, K, V projections for this head
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim
            
            with torch.no_grad():
                # Q projection
                attn.q_proj.weight.data[:, start_idx:end_idx] = 0
                if attn.q_proj.bias is not None:
                    attn.q_proj.bias.data[start_idx:end_idx] = 0
                
                # K projection
                attn.k_proj.weight.data[:, start_idx:end_idx] = 0
                if attn.k_proj.bias is not None:
                    attn.k_proj.bias.data[start_idx:end_idx] = 0
                
                # V projection
                attn.v_proj.weight.data[:, start_idx:end_idx] = 0
                if attn.v_proj.bias is not None:
                    attn.v_proj.bias.data[start_idx:end_idx] = 0
                
                # Output projection
                attn.o_proj.weight.data[start_idx:end_idx, :] = 0
        
        print(f"Pruned {len(heads_to_prune)} attention heads")
    
    def calculate_model_size(self):
        """Calculate model size in GB"""
        total_params = sum(p.numel() for p in self.model.parameters())
        # Assuming float16 (2 bytes per parameter)
        size_gb = (total_params * 2) / (1024 ** 3)
        return size_gb, total_params
    
    def validate_pruned_model(self, test_prompts=None):
        """
        Test the pruned model with sample prompts
        """
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "I love you",
                "Tell me about yourself"
            ]
        
        print("\n" + "="*60)
        print("VALIDATING PRUNED MODEL")
        print("="*60 + "\n")
        
        self.model.eval()
        
        for prompt in test_prompts:
            print(f"Prompt: {prompt}")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}\n")
        
        print("="*60 + "\n")
    
    def save_pruned_model(self, output_path="models/pruned"):
        """
        Save the pruned model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving pruned model to {output_path}...")
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        self.config.save_pretrained(output_path)
        
        # Save pruning info
        size_gb, total_params = self.calculate_model_size()
        
        info = {
            'total_parameters': int(total_params),
            'model_size_gb': float(size_gb),
            'num_layers': self.config.num_hidden_layers,
            'pruning_config': self.prune_config['pruning']
        }
        
        with open(output_path / 'pruning_info.yaml', 'w') as f:
            yaml.dump(info, f)
        
        print(f"Model saved successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {size_gb:.2f} GB")
    
    def auto_prune(self, use_analysis=True):
        """
        Automatically prune model based on analysis or config
        """
        print("\n" + "="*60)
        print("STARTING AUTOMATIC PRUNING")
        print("="*60 + "\n")
        
        # Get original size
        orig_size_gb, orig_params = self.calculate_model_size()
        print(f"Original model: {orig_params:,} parameters ({orig_size_gb:.2f} GB)")
        
        if use_analysis and Path("analysis_results.yaml").exists():
            print("\nUsing analysis results for pruning...")
            with open("analysis_results.yaml", 'r') as f:
                analysis = yaml.safe_load(f)
            
            layers_to_remove = analysis['recommendations']['layers_to_prune']
            heads_to_prune = analysis['recommendations']['heads_to_prune']
        else:
            print("\nUsing config-based pruning (run analyze.py first for better results)...")
            # Simple heuristic: remove middle layers
            total_layers = len(self.model.model.layers)
            target_reduction = self.prune_config['pruning']['target_reduction']
            num_to_remove = int(total_layers * target_reduction * 0.5)  # Conservative
            
            # Remove middle layers (keep early and late layers)
            middle_start = total_layers // 3
            middle_end = 2 * total_layers // 3
            layers_to_remove = list(range(middle_start, middle_start + num_to_remove))
            heads_to_prune = []
        
        # Prune layers
        self.prune_layers(layers_to_remove)
        
        # Prune attention heads
        if heads_to_prune:
            self.prune_attention_heads(heads_to_prune)
        
        # Get new size
        new_size_gb, new_params = self.calculate_model_size()
        reduction = (1 - new_params / orig_params) * 100
        
        print(f"\n" + "="*60)
        print("PRUNING COMPLETE")
        print("="*60)
        print(f"Original: {orig_params:,} params ({orig_size_gb:.2f} GB)")
        print(f"Pruned:   {new_params:,} params ({new_size_gb:.2f} GB)")
        print(f"Reduction: {reduction:.1f}%")
        print("="*60 + "\n")
        
        return reduction


def main():
    parser = argparse.ArgumentParser(description="Prune Llama model")
    parser.add_argument("--model", type=str, default="models/base",
                        help="Path to base model")
    parser.add_argument("--output", type=str, default="models/pruned",
                        help="Path to save pruned model")
    parser.add_argument("--config", type=str, default="configs/personality.yaml",
                        help="Path to config file")
    parser.add_argument("--validate", action="store_true",
                        help="Validate pruned model with test prompts")
    parser.add_argument("--target-reduction", type=float,
                        help="Override target reduction percentage (0-1)")
    
    args = parser.parse_args()
    
    # Initialize pruner
    pruner = StructuredPruner(args.model, args.config)
    
    # Override target reduction if specified
    if args.target_reduction:
        pruner.prune_config['pruning']['target_reduction'] = args.target_reduction
    
    # Auto prune
    reduction = pruner.auto_prune()
    
    # Validate if requested
    if args.validate:
        pruner.validate_pruned_model()
    
    # Save
    pruner.save_pruned_model(args.output)
    
    print(f"\n✅ Pruning complete! Model reduced by {reduction:.1f}%")
    print(f"📁 Saved to: {args.output}")


if __name__ == "__main__":
    main()
