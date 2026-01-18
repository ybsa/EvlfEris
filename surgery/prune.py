import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import yaml
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn

class ActivationPruner:
    def __init__(self, config_path="configs/phase1.yaml", model_path="models/base"):
        self.model_path = model_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.activations = {}
        self.masks = {}
        
    def load_model(self):
        print(f"📦 Loading {self.model_path} in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded.")

    def get_calibration_data(self):
        # Use simple list for now, can perform more complex data loading later
        return self.config['phase1']['calibration_data']['domain'] * 10 # Repeat to have enough batches

    def analyze_activations(self):
        print("🔬 Analyzing Neuron Activations on Domain Data...")
        
        # 1. Register Hooks
        handles = []
        def get_activation_hook(layer_idx):
            def hook(module, input, output):
                # MLP output is typically [batch, seq, hidden_size]
                # In Llama, the "neuron" dimension is the intermediate dimension.
                # We want to prune the INTERMEDIATE size (e.g. 11008 for 7B, less for 3B)
                # But module here is the specific Linear layer or activation?
                # Actually, we want to prune the neurons in the 'up/gate' projections.
                # Llama MLP: down(act(gate(x)) * up(x))
                # The 'neuron' is the dimension coming OUT of gate/up and INTO down.
                # So we want to inspect the input to down_proj.
                
                # Careful: The input to this hook (if attached to down_proj) is the activations we care about.
                hidden_states = input[0] # [batch, seq, intermediate_size]
                
                # Calculate importance: Sum of L1 norm over batch and sequence
                importance = hidden_states.abs().sum(dim=(0, 1))
                
                if layer_idx not in self.activations:
                    self.activations[layer_idx] = importance
                else:
                    self.activations[layer_idx] += importance
            return hook

        target_layers = self.config['phase1']['pruning']['layers_scope']
        ignored = self.config['phase1']['pruning']['ignored_layers']
        
        total_layers = len(self.model.model.layers)
        start_layer = 0
        end_layer = total_layers
        
        if target_layers == "middle":
            start_layer = int(total_layers * 0.25)
            end_layer = int(total_layers * 0.75)

        for i in range(start_layer, end_layer):
            if i in ignored or (i - total_layers) in ignored:
                continue
            # Hook the down_proj layer to see its INPUT (which is the output of the neurons)
            handles.append(self.model.model.layers[i].mlp.down_proj.register_forward_hook(get_activation_hook(i)))
            
        # 2. Forward Pass
        data = self.get_calibration_data()
        self.model.eval()
        
        with torch.no_grad():
            for text in tqdm(data, desc="Calibrating"):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                self.model(**inputs)
                
        # Remove hooks
        for h in handles:
            h.remove()
            
    def generate_masks(self):
        print("🎭 Generating Pruning Masks...")
        target_reduction = self.config['phase1']['pruning']['target_reduction']
        
        total_pruned = 0
        total_neurons = 0
        
        for layer_idx, score in self.activations.items():
            num_neurons = score.shape[0]
            num_to_prune = int(num_neurons * target_reduction)
            
            # Find threshold
            threshold = torch.kthvalue(score, num_to_prune).values.item()
            
            # Create mask (1 = keep, 0 = prune)
            mask = score >= threshold
            
            self.masks[layer_idx] = mask.cpu()
            
            pruned_count = (~mask).sum().item()
            total_pruned += pruned_count
            total_neurons += num_neurons
            
            print(f"   Layer {layer_idx}: Pruned {pruned_count}/{num_neurons} neurons")
            
        print(f"✅ Total Neurons Pruned: {total_pruned} / {total_neurons} ({total_pruned/total_neurons*100:.1f}%)")
        
        # Save masks
        torch.save(self.masks, "pruning_masks.pt")
        print("📁 Saved masks to pruning_masks.pt")

if __name__ == "__main__":
    pruner = ActivationPruner()
    pruner.load_model()
    pruner.analyze_activations()
    pruner.generate_masks()
