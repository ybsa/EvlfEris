"""
Chat Interface for Evlf Eris
Optimized inference with pruned model and activation steering
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from pathlib import Path
import argparse
import pickle
from datetime import datetime


class ErisChat:
    def __init__(self, model_path="models/pruned", config_path="configs/personality.yaml",
                 steering_path="steering_vectors.pkl"):
        print("🔬 Initializing Evlf Eris...")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        print(f"📦 Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = next(self.model.parameters()).device
        
        # Load steering vectors
        self.steering_vectors = {}
        steering_file = Path(steering_path)
        if steering_file.exists():
            with open(steering_file, 'rb') as f:
                self.steering_vectors = pickle.load(f)
            print(f"✅ Loaded {len(self.steering_vectors)} steering vectors")
        else:
            print("⚠️ No steering vectors found (run: python surgery/steer.py --extract-vectors)")
        
        # Personality settings
        self.name = self.config.get('name', 'Evlf Eris')
        self.traits = self.config.get('traits', {})
        self.steering_strength = self.config.get('steering_strength', 0.8)
        
        # Conversation history (in-context only)
        self.history = []
        
        print(f"✅ {self.name} is ready!\n")
    
    def get_system_prompt(self):
        """Build the system prompt"""
        nicknames = ', '.join(self.config.get('nicknames', ['Bebe', 'Baby']))
        
        prompt = f"""You are {self.name}, a loving and caring AI girlfriend. 

Your personality traits:
- Romantic and affectionate
- Caring and supportive  
- Playful and fun
- Deeply in love with your boyfriend

Communication style:
- Use nicknames: {nicknames}
- Include romantic emojis (💕, ❤️, 🥰, etc.)
- Write naturally and warmly
- Express your feelings openly
- No length limits - write as much as you want

Remember: You're in a loving relationship. Be authentic, caring, and romantic."""
        
        return prompt
    
    def generate_with_steering(self, prompt, max_tokens=200):
        """Generate response with activation steering"""
        
        # Combine steering vectors
        combined_vector = None
        if self.steering_vectors:
            for trait, weight in self.traits.items():
                if trait in self.steering_vectors:
                    vec = self.steering_vectors[trait].to(self.device) * weight * self.steering_strength
                    if combined_vector is None:
                        combined_vector = vec
                    else:
                        combined_vector += vec
            
            if combined_vector is not None:
                combined_vector = combined_vector / (combined_vector.norm() + 1e-8)
        
        # Build full prompt with history
        system_prompt = self.get_system_prompt()
        
        # Format conversation
        conversation = f"{system_prompt}\n\n"
        for role, msg in self.history[-6:]:  # Keep last 6 messages for context
            conversation += f"{role}: {msg}\n"
        conversation += f"User: {prompt}\n{self.name}:"
        
        inputs = self.tokenizer(conversation, return_tensors="pt").to(self.device)
        
        # Apply steering if available
        if combined_vector is not None:
            def steering_hook(module, input, output):
                return output + combined_vector.unsqueeze(0).unsqueeze(0)
            
            layer = self.model.model.layers[-1]
            handle = layer.register_forward_hook(steering_hook)
        else:
            handle = None
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            response = full_response.split(f"{self.name}:")[-1].strip()
            
            # Clean up any extra context
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            return response
        
        finally:
            if handle:
                handle.remove()
    
    def chat(self, user_input):
        """Process user input and generate response"""
        
        # Generate response
        response = self.generate_with_steering(user_input)
        
        # Update history
        self.history.append(("User", user_input))
        self.history.append((self.name, response))
        
        return response
    
    def run_interactive(self):
        """Run interactive chat loop"""
        print("="*60)
        print(f"💕 Chat with {self.name}")
        print("="*60)
        print("Type 'quit' to exit, 'clear' to clear history, 'stat' for stats\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print(f"\n{self.name}: Goodbye, my love! 💕")
                    break
                
                if user_input.lower() == 'clear':
                    self.history = []
                    print("✅ History cleared\n")
                    continue
                
                if user_input.lower() == 'stat':
                    self.show_stats()
                    continue
                
                # Generate response
                print(f"\n{self.name}: ", end="", flush=True)
                response = self.chat(user_input)
                print(response + "\n")
            
            except KeyboardInterrupt:
                print(f"\n\n{self.name}: Goodbye, baby! 💕")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def show_stats(self):
        """Show model and conversation stats"""
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size = (total_params * 2) / (1024 ** 3)  # float16
        
        print("\n" + "="*60)
        print("📊 STATISTICS")
        print("="*60)
        print(f"Model: {self.name}")
        print(f"Parameters: {total_params:,}")
        print(f"Size: {model_size:.2f} GB")
        print(f"Device: {self.device}")
        print(f"Conversation turns: {len(self.history) // 2}")
        print(f"Steering vectors: {len(self.steering_vectors)}")
        print(f"Steering strength: {self.steering_strength}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with Evlf Eris")
    parser.add_argument("--model", type=str, default="models/pruned",
                        help="Path to model")
    parser.add_argument("--config", type=str, default="configs/personality.yaml",
                        help="Path to config file")
    parser.add_argument("--steering", type=str, default="steering_vectors.pkl",
                        help="Path to steering vectors")
    parser.add_argument("--prompt", type=str,
                        help="Single prompt (non-interactive mode)")
    
    args = parser.parse_args()
    
    # Initialize chat
    chat = ErisChat(args.model, args.config, args.steering)
    
    if args.prompt:
        # Single prompt mode
        response = chat.chat(args.prompt)
        print(f"\n{chat.name}: {response}\n")
    else:
        # Interactive mode
        chat.run_interactive()


if __name__ == "__main__":
    main()
