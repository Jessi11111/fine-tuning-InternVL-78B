"""
Fine-tune InternVL model for dress classification.
Uses CSV training data with base64 images and classification attributes.

Usage:
    # First, prepare training data
    python prepare_internvl_training_data.py
    
    # Set environment variables (optional)
    export CSV_FILE="internvl_training_data.csv"
    export MODEL_NAME="OpenGVLab/InternVL3-8B-hf"  # Default: InternVL3-8B (fits 31GB GPU)
    # For larger models with quantization:
    # export MODEL_NAME="OpenGVLab/InternVL3-26B-hf"
    # export USE_8BIT="true"  # Requires: pip install bitsandbytes
    export OUTPUT_DIR="./internvl_finetuned"
    export BATCH_SIZE=1  # Use batch_size=1 to avoid OOM (use gradient_accumulation_steps for effective batch size)
    export GRADIENT_ACCUMULATION_STEPS=2  # Effective batch size = 1 * 2 = 2
    export USE_8BIT=true  # Enable 8-bit quantization to reduce memory by ~50% (recommended for 31GB GPU)
    export LEARNING_RATE=5e-5
    export NUM_EPOCHS=3
    export VAL_SPLIT=0.2
    
    # Run fine-tuning
    python finetune_internvl.py

Requirements:
    - GPU recommended (training on CPU is very slow)
    - Sufficient disk space for model weights (~150GB+ for 78B model)
    - HuggingFace transformers library with InternVL support
    - accelerate package (recommended for large models and multi-GPU):
      pip install accelerate

Note: Model size recommendations for RTX 5090 (31GB GPU):
    - InternVL3-8B-hf (default): ~16GB GPU with FP16, fits comfortably ✅
    - InternVL3-26B-hf: ~32GB GPU with FP16 (may need 8-bit quantization)
    - InternVL3-78B-hf: ~80GB GPU (too large, needs quantization or multiple GPUs)
    
    For 31GB GPU, recommended models:
      * InternVL3-8B-hf: Best fit, no quantization needed (~16GB GPU)
      * InternVL3-26B-hf with 8-bit: ~16GB GPU (USE_8BIT=true)
      * InternVL3-78B-hf with 8-bit: ~40GB GPU (still too large, not recommended)
    
    Quantization options:
      - USE_8BIT=true: Reduces memory by ~50% (requires bitsandbytes)
      - USE_4BIT=true: Reduces memory by ~75% (requires bitsandbytes, slower)
"""

import os
import json
import base64
from typing import Dict, List, Any, Optional
from io import BytesIO
from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoProcessor,
    AutoModel,
    TrainingArguments,
    Trainer
)
# Import InternVLForConditionalGeneration explicitly for fine-tuning
try:
    from transformers import InternVLForConditionalGeneration
except ImportError:
    # Fallback if direct import doesn't work
    InternVLForConditionalGeneration = None
from PIL import Image
import numpy as np

# Environment configuration
CSV_FILE = os.environ.get('CSV_FILE', 'internvl_training_data.csv')
MODEL_NAME = os.environ.get('MODEL_NAME', 'OpenGVLab/InternVL3-8B-hf')  # Changed default to 8B for 31GB GPU
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './internvl_finetuned')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '1'))  # Reduced to 1 to avoid OOM with long sequences
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '5e-5'))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', '3'))
VAL_SPLIT = float(os.environ.get('VAL_SPLIT', '0.1'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '6144'))  # Increased to 4096 to accommodate image tokens (1000-3000) + prompt + response

# Quantization settings (to reduce GPU memory usage)
USE_8BIT = os.environ.get('USE_8BIT', 'False').lower() == 'true'
USE_4BIT = os.environ.get('USE_4BIT', 'False').lower() == 'true'

# All classification attributes
ATTRIBUTES = [
    'Fabrication',
    'Neckline',
    'Sub_Neckline',
    'Silhouette',
    'Sub_Silhouette',
    'Sleeve',
    'Sub_Sleeve',
    'Waist',
    'Beaded',
    'Sequins',
    'Appliques',
    'Embroidery',
    'Lace_Coverage',
]

# Prompt template (matching classify_image_vllm.py)
PROMPT_TEMPLATE = (
    "Classify the dress features from the image. "
    "Return a flat JSON object with these exact keys: Neckline, Sub_Neckline, Silhouette, Sub_Silhouette, Fabrication, Beaded, Sequins, Appliques, Embroidery, Lace_Coverage, Sleeve, Sub_Sleeve, Waist. "
    
    "CLASSIFICATION OPTIONS: "
    "Neckline: ['Off the Shoulder', 'Square', 'Straight', 'V-Neck', 'Scoop', 'Sweetheart', 'High Neck', 'One Shoulder', 'Notch', 'Cat Eye', 'Halter'] "
    "Sub_Neckline - Only if applicable: "
    " - Off the Shoulder: ['Sweetheart Off the Shoulder', 'Straight Off the Shoulder', 'Scoop Off the Shoulder', 'Plunging Off the Shoulder', 'Notch Off the Shoulder'] "
    " - V-Neck: ['Standard V-Neck', 'Deep V-Neck'] "
    " - Notch: ['Small Notch', 'Medium Notch', 'Deep Notch'] "
    
    "Silhouette: ['Ballgown', 'A-Line', 'Fit & Flare', 'Sheath'] "
    "Sub_Silhouette - Only if applicable: "
    " - Ballgown: ['Soft Ballgown', 'Full Ballgown'] "
    " - A-Line: ['Standard A-Line', 'Modified A-Line'] "
    " - Fit & Flare: ['Mermaid', 'Trumpet', 'Standard Fit & Flare'] "
    
    "FABRICATION ATTRIBUTES: "
    "Fabrication: ['Lace', 'Mikado', 'Satin', 'Taffeta', 'Charmeuse', 'Crepe', 'Chiffon', 'Organza', 'Tulle', 'Dupioni', 'Silk', 'Jacquard', 'Organdy'] "
    "Beaded: ['Yes', 'No'] "
    "Sequins: ['Yes', 'No'] "
    "Appliques: ['Yes', 'No'] "
    "Embroidery: ['Yes', 'No'] "
    "Lace_Coverage: ['Allover', 'Top Only', 'Skirt Only', 'Sleeves Only', 'Train Only', 'Back Only', 'Mixed', 'None'] "
    
    "Sleeve: ['Strapless', 'Sleeveless', 'With Sleeve'] "
    "Sub_Sleeve - Required based on Sleeve: "
    " - Sleeveless: ['Off the Shoulder Straps', 'Spaghetti Straps', 'Medium Straps', 'Wide Straps', 'Standard Sleeveless', 'Bow Tied Straps'] "
    " - With Sleeve: ['Cap Sleeve', 'Short Sleeve', 'Long Sleeve', 'Puff Sleeve'] "
    " - Strapless: No Sub_Sleeve needed (set value to null) "
    
    "Waist: ['Natural Waist', 'Drop Waist', 'Natural Basque Waist', 'Drop Basque Waist', 'Empire Waist'] "
    
    "RULES: "
    "1. Each main feature must have exactly one value from the allowed options. "
    "2. Sub-features (Sub_Neckline, Sub_Silhouette, Sub_Sleeve) are OPTIONAL - only use if the main category has sub-options available. "
    "3. Lace_Coverage is REQUIRED - use 'None' if no lace is present. "
    "4. Return value to null for any feature that cannot be determined from the image. "
    "5. NEVER invent values outside the provided options. "
    
    "Return only valid JSON, no additional text."
)


def create_target_json(row: pd.Series) -> str:
    """
    Create target JSON string from row attributes.
    """
    target_dict = {}
    for attr in ATTRIBUTES:
        value = row.get(attr, '')
        if pd.isna(value) or value == '' or str(value).strip() == '':
            target_dict[attr] = None
        else:
            target_dict[attr] = str(value).strip()
    
    return json.dumps(target_dict, ensure_ascii=False)


def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    """
    try:
        # Handle data URI format
        if base64_str.startswith('data:'):
            base64_str = base64_str.split(',', 1)[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        # Return a blank image as fallback
        return Image.new('RGB', (224, 224), color=(255, 255, 255))


class InternVLDataset(Dataset):
    """
    Dataset for InternVL fine-tuning.
    """
    
    def __init__(
        self,
        csv_path: str,
        processor: Any,
        max_length: int = 2048  # Increased default for long prompts + image tokens
    ):
        """
        Args:
            csv_path: Path to CSV file with training data
            processor: InternVL processor
            max_length: Maximum sequence length
        """
        self.processor = processor
        self.max_length = max_length
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"📦 Loaded {len(self.df)} samples from {csv_path}")
        
        # Filter out rows without images
        self.df = self.df[self.df['image_base64'].notna()]
        self.df = self.df[self.df['image_base64'].str.len() > 100]
        print(f"✅ After filtering: {len(self.df)} samples with valid images")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Handle samples that are too long by trying multiple indices
        max_attempts = min(10, len(self.df))  # Try up to 10 different samples
        for attempt in range(max_attempts):
            try:
                actual_idx = (idx + attempt) % len(self.df)
                row = self.df.iloc[actual_idx]
                
                # Get image
                image_b64 = row['image_base64']
                image = base64_to_image(image_b64)
                
                # Create target JSON
                target_json = create_target_json(row)
                
                # Process the sample - this will raise ValueError if too long
                return self._process_sample(row, image, target_json)
                
            except ValueError as e:
                if "exceeds MAX_LENGTH" in str(e):
                    # Sample is too long, try next one
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        # All attempts failed, return a minimal valid sample
                        print(f"⚠️  Warning: Could not find valid sample after {max_attempts} attempts, using fallback")
                        return self._create_fallback_sample(row, image, target_json)
                else:
                    # Different error, re-raise
                    raise
        
        # Should not reach here
        raise ValueError("Failed to process sample")
    
    def _process_sample(self, row: pd.Series, image: Image.Image, target_json: str) -> Dict[str, torch.Tensor]:
        """
        Process a single sample. Separated for error handling.
        """
        
        # Get image token from processor (InternVL requires image placeholder in text)
        # The processor will replace this with actual image tokens
        try:
            # InternVL processor has image_token attribute
            image_token = self.processor.image_token
        except AttributeError:
            # Fallback: try to get from tokenizer
            try:
                image_token = self.processor.tokenizer.context_image_token
            except AttributeError:
                # Last resort fallback
                image_token = '<image>'
        
        # Format prompt with image placeholder
        # InternVL processor expects the image token in the text
        user_text = f"{image_token}\n{PROMPT_TEMPLATE}"
        assistant_text = target_json
        
        # Process image and text together using InternVL processor
        # InternVL processor handles image token replacement and returns processed inputs
        inputs = self.processor(
            images=image,
            text=user_text,
            return_tensors="pt",
            padding=True,
            truncation=False,  # Disable truncation to preserve image tokens
        )
        
        # Flatten tensors from processor
        # InternVL processor returns pixel_values with shape [1, num_patches, channels, height, width]
        # After squeeze(0): [num_patches, channels, height, width]
        # The model's get_image_features expects patches concatenated, which we'll handle in collate
        inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}
        
        # For InternVL fine-tuning, we need to create the full conversation
        # Format: [image_tokens] + prompt_tokens + response_tokens
        # Labels: -100 for prompt, actual token ids for response
        
        # Tokenize the assistant response separately
        assistant_tokens = self.processor.tokenizer(
            assistant_text,
            return_tensors="pt",
            padding=False,  # No padding yet
            truncation=False,  # Don't truncate - we'll handle it
            add_special_tokens=False  # Don't add BOS/EOS as we're appending
        )
        assistant_ids = assistant_tokens['input_ids'].squeeze(0) if assistant_tokens['input_ids'].dim() > 1 else assistant_tokens['input_ids']
        
        # Get current input_ids (already includes image + prompt)
        input_ids = inputs['input_ids']
        
        # CRITICAL: NEVER truncate input_ids - it contains image tokens that must match pixel_values
        # InternVL processor has already inserted image tokens into input_ids
        # Truncating input_ids would remove image tokens, causing mismatch with pixel_values
        # We can only truncate the assistant response (text only)
        
        # Append assistant response to input_ids
        # This creates the full sequence: [image_tokens] + [prompt_tokens] + [response_tokens]
        max_total_length = self.max_length  # Use MAX_LENGTH as hard limit
        
        # Check if sequence would exceed limit
        total_length = len(input_ids) + len(assistant_ids)
        if total_length > max_total_length:
            # Calculate available space for response
            # Keep ALL of input_ids (image + prompt) - never truncate it
            available_for_response = max_total_length - len(input_ids)
            
            if available_for_response > 0:
                # Truncate only the response to fit
                original_response_len = len(assistant_ids)
                assistant_ids = assistant_ids[:available_for_response]
                if len(assistant_ids) < original_response_len:
                    print(f"⚠️  Warning: Truncated response from {original_response_len} to {len(assistant_ids)} tokens (preserved all {len(input_ids)} input tokens including image tokens)")
            else:
                # input_ids itself is longer than max_total_length
                # This means image tokens + prompt is too long
                # We cannot truncate input_ids without breaking image token matching
                # Skip this sample by raising an exception that will be caught
                raise ValueError(
                    f"Sample input_ids length ({len(input_ids)}) exceeds MAX_LENGTH ({max_total_length}). "
                    f"Cannot truncate as it contains image tokens. "
                    f"Increase MAX_LENGTH or this sample will be skipped."
                )
        
        full_input_ids = torch.cat([input_ids, assistant_ids], dim=0)
        
        # Create labels: -100 (ignore) for prompt part, actual tokens for response part
        # For causal LM: model predicts next token, so labels[i] = input_ids[i+1]
        # Trainer will handle shifting automatically, but we set labels correctly
        prompt_labels = torch.full_like(input_ids, -100)  # Ignore prompt tokens
        response_labels = assistant_ids.clone()  # Learn from response tokens
        
        # Concatenate: labels should match input_ids length
        # The model will shift internally: loss is computed on predicting next token
        full_labels = torch.cat([prompt_labels, response_labels], dim=0)
        
        # Update inputs with full sequence
        inputs['input_ids'] = full_input_ids
        
        # Update attention mask to cover full sequence
        if 'attention_mask' in inputs:
            prompt_mask = inputs['attention_mask']
            if len(prompt_mask) != len(input_ids):
                # Adjust mask length if input_ids was truncated
                prompt_mask = prompt_mask[:len(input_ids)]
            response_mask = torch.ones_like(assistant_ids, dtype=prompt_mask.dtype)
            full_attention_mask = torch.cat([prompt_mask, response_mask], dim=0)
            inputs['attention_mask'] = full_attention_mask
        
        # Add labels
        inputs['labels'] = full_labels
        
        return inputs
    
    def _create_fallback_sample(self, row: pd.Series, image: Image.Image, target_json: str) -> Dict[str, torch.Tensor]:
        """
        Create a minimal fallback sample when all attempts fail.
        This should rarely be used, but prevents training from crashing.
        """
        # Use a very short prompt and minimal response
        try:
            image_token = self.processor.image_token if hasattr(self.processor, 'image_token') else '<image>'
        except:
            image_token = '<image>'
        
        # Minimal prompt
        minimal_prompt = f"{image_token}\nClassify dress features. Return JSON."
        minimal_response = '{"Fabrication":"Lace"}'
        
        inputs = self.processor(
            images=image,
            text=minimal_prompt,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}
        
        assistant_tokens = self.processor.tokenizer(
            minimal_response,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False
        )
        assistant_ids = assistant_tokens['input_ids'].squeeze(0) if assistant_tokens['input_ids'].dim() > 1 else assistant_tokens['input_ids']
        
        input_ids = inputs['input_ids']
        full_input_ids = torch.cat([input_ids, assistant_ids], dim=0)
        
        prompt_labels = torch.full_like(input_ids, -100)
        response_labels = assistant_ids.clone()
        full_labels = torch.cat([prompt_labels, response_labels], dim=0)
        
        inputs['input_ids'] = full_input_ids
        if 'attention_mask' in inputs:
            prompt_mask = inputs['attention_mask']
            response_mask = torch.ones_like(assistant_ids, dtype=prompt_mask.dtype)
            inputs['attention_mask'] = torch.cat([prompt_mask, response_mask], dim=0)
        inputs['labels'] = full_labels
        
        return inputs


def collate_fn(batch: List[Dict], processor: Any) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for InternVL.
    Handles padding for both text and images.
    """
    if not batch:
        raise ValueError("Empty batch received")
    
    # Separate pixel_values from other inputs
    pixel_values_list = []
    other_inputs = []
    
    for item in batch:
        if not isinstance(item, dict):
            continue
        item_copy = {k: v for k, v in item.items() if k != 'pixel_values'}
        if 'pixel_values' in item:
            pixel_values_list.append(item['pixel_values'])
        other_inputs.append(item_copy)
    
    if not other_inputs:
        raise ValueError("No valid inputs in batch")
    
    # Get all keys from first item
    all_keys = set()
    for item in other_inputs:
        all_keys.update(item.keys())
    
    # Pad text inputs to same length
    max_len = max(len(item.get('input_ids', torch.tensor([]))) for item in other_inputs)
    if max_len == 0:
        raise ValueError("All sequences have zero length")
    
    pad_token_id = 0
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'pad_token_id'):
        if processor.tokenizer.pad_token_id is not None:
            pad_token_id = processor.tokenizer.pad_token_id
    
    collated = {}
    
    # Handle sequence fields (input_ids, attention_mask, labels)
    for key in ['input_ids', 'attention_mask', 'labels']:
        if key not in all_keys:
            continue
        
        padded = []
        for item in other_inputs:
            seq = item.get(key)
            if seq is None:
                continue
            
            # Ensure it's a tensor
            if not isinstance(seq, torch.Tensor):
                seq = torch.tensor(seq)
            
            # Pad sequence
            if len(seq) < max_len:
                if key == 'labels':
                    # For labels, pad with -100 (ignore index)
                    padding = torch.full((max_len - len(seq),), -100, dtype=seq.dtype)
                elif key == 'input_ids':
                    padding = torch.full((max_len - len(seq),), pad_token_id, dtype=seq.dtype)
                else:  # attention_mask
                    padding = torch.zeros(max_len - len(seq), dtype=seq.dtype)
                seq = torch.cat([seq, padding])
            elif len(seq) > max_len:
                # Truncate if somehow longer (shouldn't happen, but be safe)
                seq = seq[:max_len]
            
            padded.append(seq)
        
        if padded:
            collated[key] = torch.stack(padded)
    
    # Handle other fields (stack as-is)
    for key in all_keys:
        if key in ['input_ids', 'attention_mask', 'labels', 'pixel_values']:
            continue
        
        values = [item[key] for item in other_inputs if key in item]
        if values:
            # Try to stack if all are tensors of same shape
            try:
                collated[key] = torch.stack(values)
            except (RuntimeError, TypeError):
                # If can't stack, just keep as list
                collated[key] = values
    
    # Handle pixel_values - InternVL has a special patch-based architecture
    # The processor returns patches: [num_patches, channels, height, width] per image
    # InternVL's model.get_image_features() expects all patches concatenated
    # But the vision_tower.forward() expects [batch, channels, height, width]
    # The solution: InternVL processes patches through get_image_features, not vision_tower directly
    # We concatenate all patches and let get_image_features handle it
    if pixel_values_list:
        try:
            # Concatenate all patches from all images
            # This matches how InternVL's processor works: [total_patches, channels, height, width]
            # The model's get_image_features will process this correctly
            collated['pixel_values'] = torch.cat(pixel_values_list, dim=0)
        except RuntimeError as e:
            print(f"❌ Error concatenating pixel_values: {e}")
            print(f"   Shapes: {[pv.shape for pv in pixel_values_list]}")
            raise
    
    return collated


def main():
    """
    Main training function.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  Warning: Training on CPU will be very slow. Consider using GPU.")
    
    # Load processor and model from HuggingFace
    # Note: "OpenGVLab/InternVL3-78B-hf" means OpenGVLab's model hosted on HuggingFace
    # The transformers library will automatically download from HuggingFace
    print(f"📥 Loading model from HuggingFace: {MODEL_NAME}")
    print(f"   (This is OpenGVLab's model hosted on HuggingFace platform)")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Check if accelerate is available (needed for device_map="auto")
        try:
            import accelerate
            has_accelerate = True
            print(f"✅ Found accelerate package (version: {accelerate.__version__})")
            print(f"   Will use device_map='auto' for optimal GPU memory management")
        except ImportError:
            has_accelerate = False
            print("⚠️  Warning: 'accelerate' package not found.")
            print("   device_map='auto' requires accelerate. Installing it is recommended for large models.")
            print("   Falling back to manual device placement.")
            print("   Install with: pip install accelerate")
        
        # Check quantization requirements
        quantization_config = None
        if USE_8BIT or USE_4BIT:
            try:
                from transformers import BitsAndBytesConfig
                if USE_4BIT:
                    print(f"📦 Using 4-bit quantization (reduces memory by ~75%)")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                elif USE_8BIT:
                    print(f"📦 Using 8-bit quantization (reduces memory by ~50%)")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
            except ImportError:
                print("⚠️  Warning: bitsandbytes not found. Quantization requires: pip install bitsandbytes")
                print("   Continuing without quantization...")
                quantization_config = None
        
        # InternVL models use InternVLForConditionalGeneration for fine-tuning
        # This class has the language modeling head needed for loss computation
        # AutoModel might load the base InternVLModel which doesn't compute loss
        if InternVLForConditionalGeneration is not None:
            print(f"   Loading InternVLForConditionalGeneration (has LM head for training)")
            ModelClass = InternVLForConditionalGeneration
        else:
            print(f"   Loading InternVL model (AutoModel will auto-detect)")
            ModelClass = AutoModel
        
        # Prepare loading arguments
        load_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        
        # Add quantization if requested
        if quantization_config:
            load_kwargs['quantization_config'] = quantization_config
            # With quantization, we don't set torch_dtype (quantization handles it)
        else:
            load_kwargs['torch_dtype'] = torch.float16 if device.type == 'cuda' else torch.float32
        
        # Use device_map only if accelerate is available and using GPU
        # Try device_map="auto" first, fallback to manual placement if it fails
        if device.type == 'cuda' and has_accelerate:
            print(f"   Attempting to load with device_map='auto'...")
            try:
                load_kwargs['device_map'] = "auto"
                model = ModelClass.from_pretrained(MODEL_NAME, **load_kwargs)
                print(f"   ✅ Successfully loaded with device_map='auto'")
            except (KeyError, ValueError, RuntimeError) as e:
                print(f"   ⚠️  device_map='auto' failed: {str(e)[:100]}")
                print(f"   Falling back to optimized manual loading...")
                # Fallback: Load directly to GPU with memory management
                load_kwargs['device_map'] = None
                load_kwargs['max_memory'] = {0: "30GiB", "cpu": "200GiB"}  # Limit GPU to 30GB for safety
                print(f"   Loading model with memory optimization...")
                model = ModelClass.from_pretrained(MODEL_NAME, **load_kwargs)
                # Move model to GPU manually
                if not quantization_config:  # Only move if not quantized (quantized models are already on device)
                    print(f"   Moving model to GPU...")
                    try:
                        model = model.to(device)
                        print(f"   ✅ Model moved to {device}")
                    except RuntimeError as oom_error:
                        print(f"   ❌ Out of GPU memory during model transfer!")
                        print(f"   Error: {str(oom_error)[:200]}")
                        print(f"   💡 Solutions for 31GB GPU:")
                        print(f"      1. Use InternVL3-8B-hf (recommended, fits in 31GB)")
                        print(f"      2. Use InternVL3-26B-hf with 8-bit: export USE_8BIT=true")
                        print(f"      3. Use InternVL3-8B-hf with 4-bit: export USE_4BIT=true")
                        raise
        else:
            # Manual device placement (works without accelerate)
            print(f"   Using manual device placement with memory optimization")
            load_kwargs['device_map'] = None
            load_kwargs['max_memory'] = {0: "30GiB", "cpu": "200GiB"} if device.type == 'cuda' else None
            model = ModelClass.from_pretrained(MODEL_NAME, **load_kwargs)
            # Manually move model to device (if not quantized)
            if device.type == 'cuda' and not quantization_config:
                print(f"   Moving model to GPU...")
            if not quantization_config:
                model.to(device)
        
        print("✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        
        # Ensure model is in training mode
        model.train()
        
        # Check if model supports training
        if not hasattr(model, 'forward'):
            print("⚠️  Warning: Model may not support training. Check model configuration.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Note: InternVL3 models available on HuggingFace:")
        print("   - OpenGVLab/InternVL3-78B-hf (default, ~80GB+ GPU, very large)")
        print("   - OpenGVLab/InternVL3-26B-hf (~48GB GPU)")
        print("   - OpenGVLab/InternVL3-8B-hf (~24GB GPU)")
        print("   - OpenGVLab/InternVL3-4B-hf (~16GB GPU)")
        print("   - OpenGVLab/InternVL3-1B-hf (~8GB GPU)")
        print("   - Or check HuggingFace for other InternVL3 variants")
        print("   - If model name differs, check: https://huggingface.co/OpenGVLab")
        import traceback
        traceback.print_exc()
        return
    
    # Load dataset
    print(f"📂 Loading dataset from: {CSV_FILE}")
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file not found: {CSV_FILE}")
        print("💡 Run prepare_internvl_training_data.py first to generate training data")
        return
    
    dataset = InternVLDataset(CSV_FILE, processor, MAX_LENGTH)
    
    if len(dataset) < 10:
        print(f"❌ Too few samples ({len(dataset)}). Need at least 10 samples for training.")
        return
    
    # Split dataset
    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Training arguments
    # Use bf16 instead of fp16 for better numerical stability (if available)
    # bf16 is more stable than fp16 and less prone to CUDA errors
    use_bf16 = False
    use_fp16 = False
    if device.type == 'cuda':
        # Check if bf16 is supported (Ampere+ GPUs like RTX 5090)
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("✅ Using bf16 mixed precision (more stable than fp16)")
        else:
            # Fallback to fp16, but with gradient clipping to avoid CUDA errors
            use_fp16 = True
            print("⚠️  Using fp16 mixed precision (bf16 not supported)")
    
    # Calculate effective batch size with gradient accumulation
    # If batch_size=1, use gradient_accumulation_steps=2 to get effective batch_size=2
    effective_batch_size = BATCH_SIZE
    gradient_accumulation_steps = max(1, int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', '2')))
    if BATCH_SIZE == 1:
        # With batch_size=1, use gradient accumulation to maintain effective batch size
        effective_batch_size = BATCH_SIZE * gradient_accumulation_steps
        print(f"📊 Using batch_size={BATCH_SIZE} with gradient_accumulation_steps={gradient_accumulation_steps}")
        print(f"   Effective batch size: {effective_batch_size}")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,  # Use 1 for eval to save memory
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=use_fp16,  # Use fp16 only if bf16 not available
        bf16=use_bf16,  # Prefer bf16 for stability
        dataloader_pin_memory=False,  # Disable to save memory
        remove_unused_columns=False,
        max_grad_norm=1.0,  # Gradient clipping to prevent exploding gradients
        gradient_accumulation_steps=gradient_accumulation_steps,  # Accumulate gradients to maintain effective batch size
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory (trades compute for memory)
    )
    
    # Enable gradient checkpointing on the model
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ Enabled gradient checkpointing (saves memory, slightly slower)")
    
    # Clear CUDA cache before training to free up any unused memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"🧹 Cleared CUDA cache. Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create trainer
    # Create a partial function that binds the processor to collate_fn
    # Trainer's data_collator expects a function that takes only batch as argument
    data_collator = partial(collate_fn, processor=processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("🚀 Starting training...")
    try:
        trainer.train()
        print("✅ Training completed!")
        
        # Save final model
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        print(f"💾 Model saved to: {final_model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🎉 Fine-tuning complete!")


if __name__ == "__main__":
    main()

