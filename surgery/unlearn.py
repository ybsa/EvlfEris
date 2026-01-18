import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
import yaml
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import random

class LoRAUnlearner:
    def __init__(self, config_path="configs/phase1.yaml", model_path="models/base"):
        self.model_path = model_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Datasets
        self.junk_data = [
            "The latest celebrity scandal shocked the internet today...",
            "Top 10 reality TV moments of all time...",
            "Your horoscope says you will find love soon...",
            "This viral TikTok dance is taking over...",
            "Fashion fails from the red carpet event...",
            "Gossip about the famous pop star's breakup...",
            "Click here to see the shocking photos...",
            "Influencer apologizes for the controversy...",
            "The secret diet of the supermodels revealed..."
        ] * 10 
        
        self.retain_data = self.config['phase1']['calibration_data']['domain'] * 20
        
    def load_model(self):
        print(f"📦 Loading {self.model_path} in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for kbit training (Crucial fix for "no grad" error)
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
    def setup_lora(self):
        print("🔧 Configuring LoRA Adapter for Unlearning...")
        target_layers = self.config['phase1']['unlearning']['target_layers']
        
        # Transform layer indices to string suffixes if needed, or target all and filter?
        # Standard LoRA targets module names. We can specify `layers_to_transform` in LoraConfig!
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config['phase1']['unlearning']['lora']['r'],
            lora_alpha=self.config['phase1']['unlearning']['lora']['alpha'],
            lora_dropout=self.config['phase1']['unlearning']['lora']['dropout'],
            target_modules=["gate_proj", "up_proj", "down_proj"], # MLP only
            layers_to_transform=target_layers # Targeted Layers 7-20
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
    def get_batch(self, batch_size=2):
        junk = random.sample(self.junk_data, k=batch_size)
        retain = random.sample(self.retain_data, k=batch_size)
        return junk, retain

    def train_loop(self):
        print("📉 Starting Unlearning Loop...")
        
        lr = float(self.config['phase1']['unlearning']['learning_rate'])
        steps = self.config['phase1']['unlearning']['steps']
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        kl_weight = self.config['phase1']['unlearning']['objectives']['kl_weight']
        retain_weight = self.config['phase1']['unlearning']['objectives']['retain_weight']
        
        self.model.train()
        
        for step in tqdm(range(steps), desc="Unlearning Steps"):
            optimizer.zero_grad()
            
            junk_text, retain_text = self.get_batch()
            
            # Tokenize
            junk_inputs = self.tokenizer(junk_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            retain_inputs = self.tokenizer(retain_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # 1. Junk Loss (Maximize -> Gradient Ascent)
            outputs_junk = self.model(**junk_inputs, labels=junk_inputs.input_ids)
            loss_junk = outputs_junk.loss
            
            # 2. Retain Loss (Minimize)
            outputs_retain = self.model(**retain_inputs, labels=retain_inputs.input_ids)
            loss_retain = outputs_retain.loss
            
            # 3. KL Divergence (Anchor to Base)
            # We need base model logits. 
            # With PEFT, we can disable adapters to get base output.
            all_text = junk_text + retain_text
            all_inputs = self.tokenizer(all_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with self.model.disable_adapter():
                self.model.eval()
                with torch.no_grad():
                    base_outputs = self.model(**all_inputs)
                self.model.train()
            
            # Current outputs
            current_outputs = self.model(**all_inputs)
            
            # KL(Base || Current)
            # P = Base (Target), Q = Current
            probs_base = F.softmax(base_outputs.logits, dim=-1)
            log_probs_current = F.log_softmax(current_outputs.logits, dim=-1)
            
            # Standard KLDivLoss expects input=log_probs, target=probs
            loss_kl = F.kl_div(log_probs_current, probs_base, reduction='batchmean', log_target=False)
            
            # Total Loss
            # Goal: Maximize Junk Loss (Negative Gradient), Minimize Retain, Minimize KL
            # Unlearning Loss = -Loss_Junk
            # Total = -Loss_Junk + Alpha*Loss_Retain + Beta*KL
            
            total_loss = -loss_junk + (retain_weight * loss_retain) + (kl_weight * loss_kl)
            
            total_loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                print(f"   Step {step}: Total={total_loss.item():.4f} | Junk(Ascent)={loss_junk.item():.4f} | Retain={loss_retain.item():.4f} | KL={loss_kl.item():.4f}")

        # Save Adapter
        print("💾 Saving Unlearning Adapter...")
        self.model.save_pretrained("models/unlearned_adapter")
        print("✅ Done.")

if __name__ == "__main__":
    unlearner = LoRAUnlearner()
    unlearner.load_model()
    unlearner.setup_lora()
    unlearner.train_loop()
