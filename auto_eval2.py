import json
import gc
import wandb
import torch
import psutil
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from data import (
    format_text,
    add_to_eval_table,
    initialize_eval_table
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPU 모델명 확인
gpu_name = torch.cuda.get_device_name(0)

# 시드 설정
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_cuda_cache():
    gc.collect()  # garbage collection 호출
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()
    print("Cuda Cache Cleaned")

# RAM 사용량 확인
def check_ram_usage(threshold=100):
    return psutil.virtual_memory().percent > threshold


def load_model_and_tokenizer(model_name):
    try:
        print(f"Starting to load model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.eos_token is None:
            tokenizer.eos_token = '<|endoftext|>'
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

        print(f"Successfully loaded model: {model_name}")
        return model, tokenizer
    except Exception as e:
        clear_cuda_cache()
        print(f"Failed to load model {model_name}: {e}")
        return None, None

def run_evaluation(model_name: str, fullname: str, datasets):
    
    model, tokenizer = load_model_and_tokenizer(fullname)

    for dataset_name in datasets:
      evaluate(model=model, tokenizer=tokenizer, model_name=fullname, dataset_name=dataset_name)
      if check_ram_usage():
        break

    del model
    del tokenizer
    clear_cuda_cache()


# 정답 예측 함수
def get_predictions(text, opt_num, tokenizer, model):
    model.eval()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)

    with torch.no_grad():
        start_event.record()
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        ).logits[0, -1]
        end_event.record()

    torch.cuda.synchronize()
    time_taken = start_event.elapsed_time(end_event)

    options = ['A', 'B', 'C', 'D'][:opt_num]
    option_logits = [(logits[tokenizer(f' {opt}').input_ids[-1]], opt) for opt in options]
    predicted_option = max(option_logits, key=lambda x: x[0])[1]
    return predicted_option, time_taken

def calculate_model_size(model):
    model_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1024 ** 2
    return model_size

# evaluation
def evaluate(model, tokenizer, model_name, dataset_name):
    print("Loading dataset...")
    if dataset_name == "winogrande":
        dataset = load_dataset(dataset_name, "winogrande_xl", split='validation[:1000]', trust_remote_code=True)
    elif dataset_name == "allenai/ai2_arc":
        dataset = load_dataset(dataset_name, "ARC-Easy", split='validation', trust_remote_code=True)
    elif dataset_name in ["boolq", "social_i_qa", "piqa"]:
        dataset = load_dataset(dataset_name, split='validation[:1000]', trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split='validation', trust_remote_code=True)

    print("Formatting dataset...")
    dataset = dataset.map(lambda ex: format_text(ex, dataset_name))

    ##############################
    run = wandb.init(
        project="TinyLLM_Final2",
        name=f"{model_name} in {dataset_name}",
        notes="",
        tags=[gpu_name, model_name, dataset_name],
        mode="online"
    )
    ##############################

    eval_table = initialize_eval_table(dataset_name)
    infer_times, correct = [], 0
    max_swap_usage = 0  # ✅ 최대 스왑 메모리 사용량 추적

    print("Evaluating dataset...")
    for i, data in enumerate(tqdm(dataset)):
        prediction, infer_time = get_predictions(data['text'], data['num_choices'], tokenizer, model)
        if prediction == data['gt']:
            correct += 1

        add_to_eval_table(eval_table, data, dataset_name, prediction, infer_time)
        infer_times.append(infer_time)

        # ✅ 최대 스왑 메모리 사용량 업데이트
        current_swap_usage = psutil.swap_memory().used / 1024 ** 2  # MB 단위 변환
        max_swap_usage = max(max_swap_usage, current_swap_usage)

    accuracy = correct / len(dataset)
    avg_infer_time = sum(infer_times) / len(dataset)
    gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
    model_size = calculate_model_size(model)

    # ✅ wandb에 최대 스왑 메모리 사용량 기록
    wandb.log({
        "Evaluation": eval_table,
        "Accuracy": accuracy * 100,
        "Average Inference Time (ms)": avg_infer_time,
        "GPU Memory Usage (MB)": gpu_memory,
        "Model Size (MB)": model_size,
        "Max Swap Memory Usage (MB)": max_swap_usage,  # 🔥 최대 스왑 사용량 추가
        "Dataset Name": dataset_name
    })

    print(f"GPU Memory Usage: {gpu_memory} MB")
    print(f"Max Swap Memory Usage: {max_swap_usage} MB")  # ✅ 콘솔 출력 추가
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Inference Time: {avg_infer_time} ms")
    run.finish()

if __name__ == "__main__":
    clear_cuda_cache()
    set_seed(42)
    try:
        wandb.login()
    except:
        print("Failed to log in to W&B.")

    with open("models.json", "r") as f:
        model_list = json.load(f)

    datasets = ["winogrande", "openbookqa", "allenai/ai2_arc", "social_i_qa", "boolq", "piqa"]
    torch.multiprocessing.set_start_method('spawn')

    for model_info in model_list:
        model_name = model_info["base_name"]
        model_fullnames = model_info["full_names"]

        for fullname in model_fullnames:
            print(f"Processing model: {fullname}")
            run_evaluation(model_name, fullname, datasets)

    print("All models have been processed!")

