import os
import sys
import subprocess
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("[START] Colab Environment Setup and Testing")
print("="*80)

print("\n[STEP 1] Upgrade pip and install essential tools")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "wheel", "-q"], check=False)

print("[STEP 2] Clean install core dependencies with specific versions")
print("         Installing torch...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", 
                "torch>=2.0.0", "-q"], check=False)

print("         Installing bitsandbytes...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                "bitsandbytes>=0.41.0", "-q"], check=False)

print("         Installing transformers...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                "transformers>=4.36.2", "-q"], check=False)

print("         Installing accelerate...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                "accelerate>=0.21.0", "-q"], check=False)

print("         Installing peft...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                "peft>=0.7.0", "-q"], check=False)

print("         Installing other dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", 
                "datasets", "huggingface-hub", "scikit-learn", "matplotlib", "-q"], check=False)

print("\n[STEP 3] Verify installations")
import torch
import transformers
import bitsandbytes as bnb
import accelerate
import peft

print(f"[INFO] Versions:")
print(f"       - torch: {torch.__version__}")
print(f"       - transformers: {transformers.__version__}")
print(f"       - bitsandbytes: {bnb.__version__}")
print(f"       - accelerate: {accelerate.__version__}")
print(f"       - peft: {peft.__version__}")

print("\n[STEP 4] Device Information")
if torch.cuda.is_available():
    print(f"[SUCCESS] GPU available")
    print(f"       - Device Name: {torch.cuda.get_device_name(0)}")
    print(f"       - GPU Count: {torch.cuda.device_count()}")
    print(f"       - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"       - CUDA Version: {torch.version.cuda}")
else:
    print(f"[WARNING] GPU not available, using CPU")

print("\n[STEP 5] Test 4-bit quantization model loading")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print("[INFO] Testing with phi3-mini model...")
    
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    print(f"[INFO] Loading quantization config...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    print(f"[INFO] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"[SUCCESS] Model loaded successfully!")
    print(f"[INFO] Model dtype: {model.dtype}")
    print(f"[INFO] Model device: {next(model.parameters()).device}")
    
    print(f"\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[SUCCESS] Tokenizer loaded successfully!")
    
    print(f"\n[INFO] Testing inference...")
    test_prompt = "What is 2+2?"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"[INFO] Input text: {test_prompt}")
    print(f"[INFO] Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs[0][len(input_ids[0]):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"[SUCCESS] Inference completed!")
    print(f"[INFO] Response: {response}")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    print("\n[SUCCESS] All tests passed successfully!")
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("[SUCCESS] Colab environment is ready for LoRA evaluation!")
print("="*80)
print("\nNext steps:")
print("1. Clone the repository: git clone https://github.com/caizongxun/lora.git")
print("2. Navigate to folder: cd lora")
print("3. Run evaluation: python final_test.py")
print("="*80)
