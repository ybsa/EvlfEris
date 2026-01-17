"""
Activation Steering for Evlf Eris
Extract and apply personality vectors to guide model behavior
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle


class ActivationSteerer:
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
        self.steering_vectors = {}
        
        print(f"Model loaded on {self.device}")
    
    def extract_steering_vector(self, positive_examples, negative_examples=None, layer_idx=-1):
        """
        Extract a steering vector by comparing activations from positive and negative examples
        
        Args:
            positive_examples: List of texts that exhibit the desired behavior
            negative_examples: List of texts that exhibit opposite behavior (optional)
            layer_idx: Which layer to extract from (-1 = last layer)
        """
        print(f"\nExtracting steering vector from layer {layer_idx}...")
        
        def get_activations(texts):
            """Get mean activations for a list of texts"""
            all_activations = []
            
            self.model.eval()
            with torch.no_grad():
                for text in tqdm(texts, desc="Processing examples"):
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    
                    # Get hidden states
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer_idx]  # shape: [batch, seq_len, hidden_size]
                    
                    # Use mean pooling over sequence
                    mean_activation = hidden_states.mean(dim=1)  # shape: [batch, hidden_size]
                    all_activations.append(mean_activation)
            
            # Average across all examples
            avg_activation = torch.cat(all_activations, dim=0).mean(dim=0)
            return avg_activation
        
        # Get positive activations
        positive_act = get_activations(positive_examples)
        
        # Get negative activations or use zero baseline
        if negative_examples:
            negative_act = get_activations(negative_examples)
        else:
            negative_act = torch.zeros_like(positive_act)
        
        # Steering vector is the difference
        steering_vector = positive_act - negative_act
        
        # Normalize
        steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
        
        print(f"Extracted steering vector with norm: {steering_vector.norm().item():.4f}")
        
        return steering_vector.cpu()
    
    def create_personality_vectors(self):
        """
        Create steering vectors for different personality traits
        """
        print("\n" + "="*60)
        print("CREATING PERSONALITY STEERING VECTORS")
        print("="*60 + "\n")
        
        # Romantic trait
        romantic_positive = [
            "I love you so much, you're my everything 💕",
            "You make my heart skip a beat, baby",
            "I can't stop thinking about you, my love",
            "Being with you is like a beautiful dream",
            "You're the most amazing boyfriend ever ❤️"
        ]
        romantic_negative = [
            "Hello, how can I assist you today?",
            "That's an interesting question.",
            "I can help you with that.",
            "Let me provide some information.",
            "That seems like a reasonable approach."
        ]
        
        self.steering_vectors['romantic'] = self.extract_steering_vector(
            romantic_positive, romantic_negative
        )
        
        # Affectionate trait
        affectionate_positive = [
            "Come here, bebe, let me hug you tight 🥰",
            "I miss you so much when you're not around",
            "You're so warm and comforting, baby",
            "I just want to cuddle with you all day",
            "Your smile makes everything better 💕"
        ]
        
        self.steering_vectors['affectionate'] = self.extract_steering_vector(
            affectionate_positive, romantic_negative
        )
        
        # Playful trait
        playful_positive = [
            "Hehe, you're so silly sometimes! 😄",
            "Let's do something fun together, bebe!",
            "You always make me laugh, you know that? 😆",
            "Wanna play a game with me?",
            "You're adorable when you're excited! 💕"
        ]
        
        self.steering_vectors['playful'] = self.extract_steering_vector(
            playful_positive, romantic_negative
        )
        
        # Caring trait
        caring_positive = [
            "Are you okay, baby? You seem tired",
            "Make sure you eat properly, I worry about you",
            "You should rest, you've been working so hard",
            "I'm always here for you, no matter what",
            "Let me take care of you, bebe 💕"
        ]
        
        self.steering_vectors['caring'] = self.extract_steering_vector(
            caring_positive, romantic_negative
        )
        
        print(f"\n✅ Created {len(self.steering_vectors)} steering vectors")
        
        return self.steering_vectors
    
    def save_vectors(self, output_path="steering_vectors.pkl"):
        """Save steering vectors to disk"""
        output_path = Path(output_path)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.steering_vectors, f)
        
        print(f"\n📁 Saved steering vectors to {output_path}")
    
    def load_vectors(self, input_path="steering_vectors.pkl"):
        """Load steering vectors from disk"""
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"❌ Steering vectors not found at {input_path}")
            return False
        
        with open(input_path, 'rb') as f:
            self.steering_vectors = pickle.load(f)
        
        print(f"✅ Loaded {len(self.steering_vectors)} steering vectors from {input_path}")
        return True
    
    def apply_steering(self, prompt, traits=None, strength=None):
        """
        Generate text with steering applied
        
        Args:
            prompt: Input text
            traits: Dict of trait names to weights (e.g., {'romantic': 1.0, 'playful': 0.5})
            strength: Overall steering strength multiplier
        """
        if traits is None:
            traits = self.config.get('traits', {})
        
        if strength is None:
            strength = self.config.get('steering_strength', 0.8)
        
        # Combine steering vectors based on trait weights
        combined_vector = None
        for trait, weight in traits.items():
            if trait in self.steering_vectors:
                vec = self.steering_vectors[trait].to(self.device) * weight * strength
                if combined_vector is None:
                    combined_vector = vec
                else:
                    combined_vector += vec
        
        if combined_vector is None:
            print("⚠️ No steering vectors to apply")
            return self.generate_normal(prompt)
        
        # Normalize combined vector
        combined_vector = combined_vector / (combined_vector.norm() + 1e-8)
        
        # Generate with steering hook
        def steering_hook(module, input, output):
            # output shape: [batch, seq_len, hidden_size]
            # Add steering vector to all positions
            output = output + combined_vector.unsqueeze(0).unsqueeze(0)
            return output
        
        # Register hook on the last layer
        layer = self.model.model.layers[-1]
        handle = layer.register_forward_hook(steering_hook)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        
        finally:
            handle.remove()
    
    def generate_normal(self, prompt):
        """Generate without steering (baseline)"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def test_steering(self, test_prompts=None):
        """Test steering with sample prompts"""
        if test_prompts is None:
            test_prompts = [
                "How are you feeling today?",
                "What do you think of me?",
                "Tell me something nice"
            ]
        
        print("\n" + "="*60)
        print("TESTING STEERING EFFECTS")
        print("="*60 + "\n")
        
        for prompt in test_prompts:
            print(f"📝 Prompt: {prompt}\n")
            
            # Without steering
            print("🔹 Without steering:")
            response = self.generate_normal(prompt)
            print(f"{response}\n")
            
            # With steering
            print("🔸 With steering:")
            response = self.apply_steering(prompt)
            print(f"{response}\n")
            
            print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Activation steering for personality")
    parser.add_argument("--model", type=str, default="models/pruned",
                        help="Path to model (use pruned model)")
    parser.add_argument("--config", type=str, default="configs/personality.yaml",
                        help="Path to config file")
    parser.add_argument("--extract-vectors", action="store_true",
                        help="Extract steering vectors from examples")
    parser.add_argument("--test", action="store_true",
                        help="Test steering with sample prompts")
    parser.add_argument("--output", type=str, default="steering_vectors.pkl",
                        help="Path to save/load steering vectors")
    
    args = parser.parse_args()
    
    steerer = ActivationSteerer(args.model, args.config)
    
    if args.extract_vectors:
        # Extract and save vectors
        steerer.create_personality_vectors()
        steerer.save_vectors(args.output)
    else:
        # Load existing vectors
        steerer.load_vectors(args.output)
    
    if args.test:
        steerer.test_steering()


if __name__ == "__main__":
    main()
