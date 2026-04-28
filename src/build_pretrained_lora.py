import json
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "Qwen/Qwen-14B"
lora_path = "/home/user_hdi1/project_smoke/shyoon_ragmed/models/2.AI학습모델파일/qwen_14b_essential_lora_cot_v2/checkpoint-32000"

# 1. 먼저 PeftConfig를 불러와서 target_modules를 강제로 수정합니다.
config = PeftConfig.from_pretrained(lora_path)
print(f"기존 타겟 모듈: {config.target_modules}")

# Qwen-14B (1세대)에 맞는 레이어 이름으로 교체
config.target_modules = ["c_attn", "c_proj"] 

# 2. 베이스 모델 로드
print("모델 로드 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype="auto", 
    device_map="cpu",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

# 3. 수정된 config와 함께 LoRA 연결
print("LoRA 어댑터 연결 및 병합 중...")
# config를 직접 넘겨주어 mismatch 해결
model = PeftModel.from_pretrained(base_model, lora_path, config=config)
merged_model = model.merge_and_unload()

# 4. 저장
save_path = "/home/user_hdi1/project_smoke/shyoon_ragmed/models/lora_merged"
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ 성공적으로 병합되었습니다!")