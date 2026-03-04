import os
import sys
import torch
from torchvision.transforms.functional import to_pil_image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.read_data import TaskReader, compute_task_stats, EvoRealDataset
from scripts.Evo1_server import load_model_and_normalizer

def main():
    ckpt_dir = "/mnt/data_ssd/zhoufang/code/evo-fast/Evo_1/checkpoints/real_stage1_only/step_6000" 
    data_path = "/mnt/data_ssd/zhoufang/code/evo-fast/Evo_1/dataset/real_data/task_1"
    prompt = "pick and place"

    print(f"\n[1] Loading model and dataset...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)
    model.eval()
    task_reader = TaskReader(data_path)
    norm_stats = compute_task_stats(task_reader)
    dataset = EvoRealDataset(task_reader, norm_stats, prompt=prompt, horizon=16, max_state_dim=7, max_action_dim=7)
    sample = dataset[0]
    
    print(f"\n[2] Saving Images...")
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_images"))
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(sample["images"].shape[0]):
        try:
            # 加入反归一化，防止全黑
            img_tensor = sample["images"][i].clone().float()
            img_min, img_max = img_tensor.min(), img_tensor.max()
            img_tensor = (img_tensor - img_min) / (img_max - img_min + 1e-8)
            
            img_pil = to_pil_image(img_tensor)
            img_path = os.path.join(save_dir, f"view_{i}_fixed.png")
            img_pil.save(img_path)
            print(f"✅ 成功保存可视图片至: {img_path}")
        except Exception as e:
            print(f"❌ 保存图片失败: {e}")

    print(f"\n[3] Running Inference...")
    images_input = sample["images"].to("cuda") 
    image_mask_input = sample["image_mask"].to("cuda")
    state_input = sample["state"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    action_mask_input = sample["action_mask"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_action, _, _ = model.run_inference(images=images_input, image_mask=image_mask_input, prompt=prompt, state_input=state_input, action_mask=action_mask_input, steps=10)
    
    pred_action = pred_action.squeeze(0).float().cpu().numpy()
    gt_action = sample["action"].float().numpy()
    
    s_min, s_max = norm_stats["pose_min"][:7], norm_stats["pose_max"][:7]
    def denorm(val, s_min, s_max): return (val + 1.0) / 2.0 * (s_max - s_min + 1e-8) + s_min

    print(f"\n[4] Results:")
    print(f"GT (Raw)  : {denorm(gt_action[0][:7], s_min, s_max)}")
    print(f"Pred(Raw) : {denorm(pred_action[:7], s_min, s_max)}\n")

if __name__ == "__main__":
    main()
