# from transformers import AutoModel, AutoTokenizer

# model_name = "meta-llama/Llama-2-7b-hf"

# # 모델과 토크나이저 다운로드
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # 원하는 경로로 저장
# save_directory = "./models/mymodel"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)

# print(f"모델과 토크나이저를 {save_directory}에 저장했습니다!")

import os
from huggingface_hub import snapshot_download

MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_NAME = MODEL_ID.split('/')[-1] # Orion-14B-Chat
data_path = "./models/" + MODEL_NAME
print(data_path)

snapshot_download(repo_id=MODEL_ID, local_dir=data_path, local_dir_use_symlinks=False, revision="main")