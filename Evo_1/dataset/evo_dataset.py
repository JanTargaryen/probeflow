import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np

# from read_data import TaskReader, compute_task_stats

class EvoRealDataset(Dataset):
    def __init__(
        self, 
        task_reader, 
        norm_stats: dict,
        prompt: str,
        horizon: int = 16, 
        max_state_dim: int = 24, 
        max_action_dim: int = 24, 
        max_views: int = 3,
        embodiment_id: int = 0
    ):
        self.reader = task_reader
        self.prompt = prompt
        self.horizon = horizon
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_views = max_views
        self.embodiment_id = embodiment_id
        
        self.state_min = norm_stats["pose_min"]
        self.state_max = norm_stats["pose_max"]
        
        self.index_map = []
        for ep_idx in range(len(self.reader)):
            ep = self.reader.get_episode(ep_idx)
            # Ensure enough steps for the horizon window
            for step_idx in range(len(ep) - self.horizon + 1):
                self.index_map.append((ep_idx, step_idx))
                
        self.transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.index_map)
        
    def _pad_tensor(self, source_tensor: torch.Tensor, max_dim: int):
        source_dim = source_tensor.shape[-1]
        padded_shape = (*source_tensor.shape[:-1], max_dim)
        
        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype)
        mask = torch.zeros(padded_shape, dtype=torch.bool)
        
        data_slice = (..., slice(0, source_dim))
        padded_tensor[data_slice] = source_tensor
        mask[data_slice] = True
        
        return padded_tensor, mask

    def __getitem__(self, idx):
        ep_idx, step_idx = self.index_map[idx]
        episode = self.reader.get_episode(ep_idx)
        step_data = episode.get_step_data(step_idx)
        
        images = []
        if step_data["rgb"] is not None:
            img_pil = Image.fromarray(step_data["rgb"])
            images.append(self.transform(img_pil))
            
        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        
        while len(images) < self.max_views:
            images.append(torch.zeros(3, 448, 448))
        images = torch.stack(images)

        # 1. State: current step pose
        raw_state = step_data["pose"] 
        state = torch.tensor(raw_state, dtype=torch.float32)
        
        s_min = torch.tensor(self.state_min, dtype=torch.float32)
        s_max = torch.tensor(self.state_max, dtype=torch.float32)
        state = 2 * (state - s_min) / (s_max - s_min + 1e-8) - 1
        state = torch.clamp(state, -1.0, 1.0)
        state_padded, state_mask = self._pad_tensor(state, self.max_state_dim)

        # 2. Action: horizon sequence starting from current step
        action_seq = []
        for t in range(step_idx, step_idx + self.horizon):
            future_step = episode.get_step_data(t)
            action_seq.append(future_step["pose"])
            
        action = torch.tensor(np.stack(action_seq), dtype=torch.float32)
        action = 2 * (action - s_min.unsqueeze(0)) / (s_max.unsqueeze(0) - s_min.unsqueeze(0) + 1e-8) - 1
        action = torch.clamp(action, -1.0, 1.0)
        action_padded, action_mask = self._pad_tensor(action, self.max_action_dim)

        return {
            "prompt": self.prompt,
            "images": images,
            "image_mask": image_mask,
            "state": state_padded.to(dtype=torch.bfloat16),
            "state_mask": state_mask,
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "embodiment_id": torch.tensor(self.embodiment_id, dtype=torch.long)
        }