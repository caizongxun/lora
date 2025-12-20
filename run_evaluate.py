import os
import subprocess
import sys

os.chdir('/content/lora')

print("Pulling latest code...\n")
subprocess.run(['git', 'pull', 'origin', 'main'], check=False)

print("\nRunning evaluate_working.py with max_samples=3...\n")
print("="*80)

result = subprocess.run([
    sys.executable,
    'evaluate_working.py',
    '--hf_model_id', 'zongowo111/phi3-lora-gsm8k-commonsense',
    '--max_samples', '3'
])

print("="*80)
print(f"\nProcess exit code: {result.returncode}")
