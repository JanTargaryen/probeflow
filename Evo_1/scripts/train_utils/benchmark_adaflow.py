import os
import sys
import torch
import numpy as np
from tqdm import tqdm
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 预防性保留
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.read_data import TaskReader, compute_task_stats, EvoRealDataset
from scripts.Evo1_server import load_model_and_normalizer
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    return {"images": images, "states": states, "action_mask": action_mask, "image_masks": image_masks}

def main():
    ckpt_dir = "/mnt/data_ssd/zhoufang/code/evo-fast/Evo_1/checkpoints/real_stage1_only/step_6000"
    data_path = "/mnt/data_ssd/zhoufang/code/evo-fast/Evo_1/dataset/real_data/task_1"
    prompt = "robot grasping task"
    
    FIXED_STEPS = 10     
    NUM_SAMPLES = 50     
    WARMUP_RUNS = 5     

    print("\n[1] Loading Model and Dataset for Benchmarking...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)
    model.eval()

    task_reader = TaskReader(data_path)
    norm_stats = {"pose_min": normalizer.state_min.cpu().numpy(), "pose_max": normalizer.state_max.cpu().numpy()}
    dataset = EvoRealDataset(task_reader, norm_stats, prompt=prompt, horizon=16, max_state_dim=7, max_action_dim=7)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    print("\n[2] Warming up GPU (CUDA graphs/kernels)...")
    sample = next(iter(dataloader))
    images_in = sample["images"][0].to("cuda")
    image_mask_in = sample["image_masks"][0].to("cuda")
    state_in = sample["states"].to("cuda", dtype=torch.bfloat16)
    action_mask_in = sample["action_mask"].to("cuda", dtype=torch.bfloat16)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(WARMUP_RUNS):
            model.run_inference(images=images_in, image_mask=image_mask_in, prompt=prompt, 
                                state_input=state_in, action_mask=action_mask_in, steps=FIXED_STEPS)

    print(f"\n[3] Running Benchmark on {NUM_SAMPLES} samples...")
    fixed_latencies = []
    adaptive_latencies = []
    adaptive_steps_list = []
    sim_scores = []

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i, batch in enumerate(tqdm(dataloader, total=NUM_SAMPLES)):
            if i >= NUM_SAMPLES: break
            
            imgs = batch["images"][0].to("cuda")
            i_mask = batch["image_masks"][0].to("cuda")
            st = batch["states"].to("cuda", dtype=torch.bfloat16)
            a_mask = batch["action_mask"].to("cuda", dtype=torch.bfloat16)

            # --- Test 1: Fixed Steps ---
            torch.cuda.synchronize()
            starter.record()
            model.run_inference(images=imgs, image_mask=i_mask, prompt=prompt, 
                                state_input=st, action_mask=a_mask, steps=FIXED_STEPS)
            ender.record()
            torch.cuda.synchronize()
            fixed_latencies.append(starter.elapsed_time(ender))

            # --- Test 2: Adaptive Steps (AdaFlow) ---
            torch.cuda.synchronize()
            starter.record()
            _, _, meta = model.run_inference(images=imgs, image_mask=i_mask, prompt=prompt, 
                                             state_input=st, action_mask=a_mask, steps=None)
            ender.record()
            torch.cuda.synchronize()
            adaptive_latencies.append(starter.elapsed_time(ender))
            
            adaptive_steps_list.append(meta.get("steps", 0))
            sim_scores.append(meta.get("sim", 0.0))

    # --- Print Benchmark Report ---
    avg_fixed_lat = np.mean(fixed_latencies)
    avg_adapt_lat = np.mean(adaptive_latencies)
    avg_adapt_steps = np.mean(adaptive_steps_list)
    speedup = (avg_fixed_lat - avg_adapt_lat) / avg_fixed_lat * 100

    print("\n" + "="*50)
    print("🚀 AdaFlow Benchmark Report (Batch Size = 1)")
    print("="*50)
    print(f"Fixed Steps Baseline ({FIXED_STEPS} steps):")
    print(f"  - Average Latency : {avg_fixed_lat:.2f} ms")
    print("-" * 50)
    print(f"AdaFlow (Adaptive):")
    print(f"  - Average Steps   : {avg_adapt_steps:.2f} (Min: {min(adaptive_steps_list)}, Max: {max(adaptive_steps_list)})")
    print(f"  - Average Latency : {avg_adapt_lat:.2f} ms")
    print(f"  - Avg Cosine Sim  : {np.mean(sim_scores):.4f}")
    print("-" * 50)
    print(f"🏆 Latency Reduction: {speedup:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
