"""
Knowledge Editing for Evlf Eris
Embed facts directly into model weights using ROME (Rank-One Model Editing)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm


class KnowledgeEditor:
    def __init__(self, model_path, config_path="configs/personality.yaml"):
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Need full precision for editing
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
    
    def locate_knowledge_layer(self, subject, expected_fact):
        """
        Locate which layer stores knowledge about a subject
        Uses causal tracing to find the most important layer
        """
        print(f"\nLocating knowledge layer for: '{subject}'")
        
        prompt = f"{subject} is"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get baseline output
        with torch.no_grad():
            baseline_outputs = self.model(**inputs, output_hidden_states=True)
            baseline_logits = baseline_outputs.logits[0, -1, :]
        
        # Expected token
        expected_tokens = self.tokenizer(expected_fact, add_special_tokens=False).input_ids
        expected_token_id = expected_tokens[0] if expected_tokens else None
        
        if expected_token_id is None:
            print("⚠️ Could not tokenize expected fact")
            return len(self.model.model.layers) // 2  # Default to middle layer
        
        baseline_prob = torch.softmax(baseline_logits, dim=0)[expected_token_id].item()
        print(f"Baseline probability for expected token: {baseline_prob:.4f}")
        
        # Find which layer has most influence
        num_layers = len(self.model.model.layers)
        layer_effects = []
        
        for layer_idx in tqdm(range(num_layers), desc="Tracing layers"):
            # This is a simplified version - full ROME uses more sophisticated causal tracing
            # For now, we'll use gradient-based importance
            self.model.zero_grad()
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0, -1, expected_token_id]
            
            logits.backward()
            
            # Get gradient magnitude for this layer
            layer = self.model.model.layers[layer_idx]
            grad_magnitude = 0
            for param in layer.parameters():
                if param.grad is not None:
                    grad_magnitude += param.grad.abs().sum().item()
            
            layer_effects.append(grad_magnitude)
        
        # Find layer with highest gradient magnitude
        best_layer = int(np.argmax(layer_effects))
        print(f"✅ Found most influential layer: {best_layer}")
        
        return best_layer
    
    def edit_fact(self, subject, new_fact, layer_idx=None):
        """
        Edit knowledge using a simplified ROME approach
        
        This is a simplified version - proper ROME requires:
        1. Computing covariance statistics over many prompts
        2. Precise rank-one updates to weight matrices
        
        For our use case, we'll use a simpler fine-tuning approach
        """
        if layer_idx is None:
            layer_idx = self.locate_knowledge_layer(subject, new_fact)
        
        print(f"\nEditing fact: '{subject}' -> '{new_fact}'")
        print(f"Target layer: {layer_idx}")
        
        # Create training example
        prompt = f"{subject}"
        full_text = f"{subject} {new_fact}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        labels = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.device)
        
        # Fine-tune just the target layer's MLP
        target_layer = self.model.model.layers[layer_idx]
        
        # Freeze all parameters except target layer MLP
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in target_layer.mlp.parameters():
            param.requires_grad = True
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in target_layer.mlp.parameters() if p.requires_grad],
            lr=1e-5
        )
        
        # Train for a few steps
        num_steps = 20
        self.model.train()
        
        for step in tqdm(range(num_steps), desc="Editing"):
            optimizer.zero_grad()
            
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                print(f"  Step {step}: Loss = {loss.item():.4f}")
        
        # Re-freeze
        for param in target_layer.mlp.parameters():
            param.requires_grad = False
        
        self.model.eval()
        print(f"✅ Fact edited successfully")
    
    def embed_personality_facts(self):
        """
        Embed all personality facts from config
        """
        facts = self.config.get('embedded_facts', [])
        
        if not facts:
            print("⚠️ No facts to embed in config")
            return
        
        print("\n" + "="*60)
        print(f"EMBEDDING {len(facts)} PERSONALITY FACTS")
        print("="*60 + "\n")
        
        for fact_info in facts:
            subject = fact_info['subject']
            fact = fact_info['fact']
            
            self.edit_fact(subject, fact)
        
        print("\n✅ All facts embedded!")
    
    def verify_facts(self):
        """
        Verify that embedded facts are recalled correctly
        """
        facts = self.config.get('embedded_facts', [])
        
        print("\n" + "="*60)
        print("VERIFYING EMBEDDED FACTS")
        print("="*60 + "\n")
        
        self.model.eval()
        
        for fact_info in facts:
            subject = fact_info['subject']
            expected = fact_info['fact']
            
            # Generate response
            inputs = self.tokenizer(subject, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,  # Deterministic for verification
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Subject: {subject}")
            print(f"Expected: {expected}")
            print(f"Got: {response}")
            
            # Check if key words from expected fact appear in response
            key_words = set(expected.lower().split())
            response_words = set(response.lower().split())
            overlap = key_words & response_words
            
            if len(overlap) / len(key_words) > 0.5:
                print("✅ PASS\n")
            else:
                print("⚠️ PARTIAL or FAIL\n")
    
    def save_edited_model(self, output_path="models/pruned"):
        """Save model with embedded facts"""
        output_path = Path(output_path)
        
        print(f"\nSaving edited model to {output_path}...")
        
        # Convert back to float16 for efficiency
        self.model = self.model.half()
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        print("✅ Model saved!")


def main():
    parser = argparse.ArgumentParser(description="Knowledge editing for personality")
    parser.add_argument("--model", type=str, default="models/pruned",
                        help="Path to model")
    parser.add_argument("--config", type=str, default="configs/personality.yaml",
                        help="Path to config file")
    parser.add_argument("--embed", action="store_true",
                        help="Embed facts from config")
    parser.add_argument("--verify", action="store_true",
                        help="Verify embedded facts")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save edited model (default: same as input)")
    
    args = parser.parse_args()
    
    editor = KnowledgeEditor(args.model, args.config)
    
    if args.embed:
        editor.embed_personality_facts()
        
        # Save model
        output_path = args.output if args.output else args.model
        editor.save_edited_model(output_path)
    
    if args.verify:
        editor.verify_facts()


if __name__ == "__main__":
    main()
