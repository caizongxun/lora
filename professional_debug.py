#!/usr/bin/env python3
"""
Professional Debugging Tool for LoRA Evaluation
Captures comprehensive system information, dependencies, and error traces

Usage:
    python professional_debug.py

Output:
    - eval_debug_complete.log (all console output)
    - eval_debug_summary.txt (summary of findings)
    - eval_debug_full.txt (detailed analysis)
"""

import sys
import os
import platform
import json
import traceback
from datetime import datetime
from io import StringIO
import subprocess

# ============================================================================
# SETUP LOGGING
# ============================================================================

class DualLogger:
    """Write to both console and file simultaneously"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.file = open(log_file, 'w')
    
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
        self.file.flush()
    
    def flush(self):
        pass
    
    def close(self):
        self.file.close()

# Start logging
log_file = '/content/eval_debug_complete.log'
original_stdout = sys.stdout
sys.stdout = DualLogger(log_file)

print("\n" + "="*100)
print("🔍 PROFESSIONAL DEBUGGING TOOL FOR LORA EVALUATION")
print("="*100)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}")
print()

# ============================================================================
# SECTION 1: SYSTEM INFORMATION
# ============================================================================

print("\n" + "="*100)
print("SECTION 1: SYSTEM INFORMATION")
print("="*100 + "\n")

print(f"Platform: {platform.platform()}")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"User: {os.environ.get('USER', 'unknown')}")
print(f"Home: {os.environ.get('HOME', 'unknown')}")

# ============================================================================
# SECTION 2: ENVIRONMENT CHECKS
# ============================================================================

print("\n" + "="*100)
print("SECTION 2: ENVIRONMENT CHECKS")
print("="*100 + "\n")

# Check GPU
print("[GPU CHECK]")
try:
    import torch
    print(f"✅ PyTorch imported")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"   Current memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
    else:
        print("   ⚠️  WARNING: No GPU available!")
except Exception as e:
    print(f"❌ PyTorch error: {e}")
    traceback.print_exc()

# ============================================================================
# SECTION 3: PACKAGE VERSIONS
# ============================================================================

print("\n" + "="*100)
print("SECTION 3: PACKAGE VERSIONS")
print("="*100 + "\n")

packages_to_check = [
    'torch',
    'transformers',
    'peft',
    'datasets',
    'bitsandbytes',
    'accelerate',
    'safetensors'
]

versions = {}
for pkg in packages_to_check:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        versions[pkg] = version
        print(f"✅ {pkg:<20} {version}")
    except ImportError:
        versions[pkg] = 'NOT INSTALLED'
        print(f"❌ {pkg:<20} NOT INSTALLED")
    except Exception as e:
        versions[pkg] = f'ERROR: {e}'
        print(f"⚠️  {pkg:<20} ERROR: {e}")

# ============================================================================
# SECTION 4: IMPORT TESTS
# ============================================================================

print("\n" + "="*100)
print("SECTION 4: DETAILED IMPORT TESTS")
print("="*100 + "\n")

def test_import(module_name, items=None):
    """Test importing a module and specific items"""
    print(f"[TEST] Importing {module_name}...")
    try:
        mod = __import__(module_name, fromlist=items or [])
        print(f"✅ {module_name} imported successfully")
        
        if items:
            for item in items:
                try:
                    getattr(mod, item)
                    print(f"   ✅ {item} available")
                except AttributeError:
                    print(f"   ❌ {item} NOT FOUND")
        return True
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        print(f"   Traceback:")
        for line in traceback.format_exc().split('\n'):
            if line:
                print(f"   {line}")
        return False

test_import('torch')
test_import('transformers', ['AutoModelForCausalLM', 'AutoTokenizer', 'BitsAndBytesConfig'])
test_import('peft', ['PeftModel', 'LoraConfig', 'get_peft_model'])
test_import('datasets', ['load_dataset'])
test_import('bitsandbytes')
test_import('accelerate')

# ============================================================================
# SECTION 5: MODEL LOADING TEST
# ============================================================================

print("\n" + "="*100)
print("SECTION 5: MODEL LOADING TEST")
print("="*100 + "\n")

try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    
    print("[STEP 1] Creating BitsAndBytesConfig...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    print("✅ Config created successfully")
    print(f"   Config type: {type(quantization_config)}")
    print(f"   Config: {quantization_config}")
    
    print("\n[STEP 2] Attempting to load base model...")
    print("   Model: microsoft/Phi-3-mini-4k-instruct")
    print("   device_map: 'cuda'")
    print("   (This may take 2-3 minutes...)")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=quantization_config,
        device_map="cuda",
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully!")
    print(f"   Model type: {type(base_model).__name__}")
    print(f"   Model dtype: {base_model.dtype}")
    print(f"   Model device: {next(base_model.parameters()).device if list(base_model.parameters()) else 'N/A'}")
    
    # Test inference
    print("\n[STEP 3] Testing tokenizer and inference...")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")
    
    prompt = "<|user|>\nWhat is 2+2?<|end|>\n<|assistant|>\n"
    print(f"\n   Tokenizing: '{prompt[:50]}...'")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    print(f"   ✅ Input IDs shape: {inputs['input_ids'].shape}")
    
    print("\n   Moving inputs to CUDA...")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    print(f"   ✅ Inputs moved to GPU")
    
    print("\n   Running inference...")
    with torch.no_grad():
        outputs = base_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=64,
            pad_token_id=tokenizer.eos_token_id
        )
    print(f"   ✅ Generation successful")
    print(f"   Output shape: {outputs.shape}")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Response: {response[len(prompt):].strip()}")
    
    print("\n✅ MODEL LOADING TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ ERROR DURING MODEL LOADING TEST")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"\nFull traceback:")
    print(traceback.format_exc())

# ============================================================================
# SECTION 6: LORA LOADING TEST
# ============================================================================

print("\n" + "="*100)
print("SECTION 6: LORA LOADING TEST")
print("="*100 + "\n")

try:
    from peft import PeftModel
    
    print("[TEST] Loading LoRA weights...")
    print("   LoRA model: zongowo111/phi3-lora-gsm8k-commonsense")
    
    lora_model = PeftModel.from_pretrained(
        base_model,
        "zongowo111/phi3-lora-gsm8k-commonsense",
        is_trainable=False
    )
    
    print("✅ LoRA loaded successfully!")
    print(f"   Model type: {type(lora_model).__name__}")
    
except Exception as e:
    print(f"❌ ERROR LOADING LORA")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"\nFull traceback:")
    print(traceback.format_exc())

# ============================================================================
# SECTION 7: SUMMARY
# ============================================================================

print("\n" + "="*100)
print("SECTION 7: SUMMARY & RECOMMENDATIONS")
print("="*100 + "\n")

summary = {
    "timestamp": datetime.now().isoformat(),
    "python_version": sys.version,
    "platform": platform.platform(),
    "packages": versions,
    "cuda_available": torch.cuda.is_available() if 'torch' in versions and versions['torch'] != 'NOT INSTALLED' else False,
    "model_loading_status": "UNKNOWN"
}

print("Package Status:")
for pkg, version in versions.items():
    status = "✅" if version not in ['NOT INSTALLED'] and not str(version).startswith('ERROR') else "❌"
    print(f"  {status} {pkg}: {version}")

print("\n" + "="*100)
print("END OF DEBUG REPORT")
print("="*100)

if hasattr(sys.stdout, 'close'):
    sys.stdout.close()
    sys.stdout = original_stdout

print(f"\n✅ Debug report saved to: {log_file}")
print("\nTo analyze the complete report:")
print(f"  cat {log_file}")
