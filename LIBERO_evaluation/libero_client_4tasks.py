import asyncio
import websockets
import numpy as np
import json
import pathlib
import os
import logging
import math
import imageio
import random
import datetime
import argparse

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["MUJOCO_GL"] = "osmes"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]


######################################
class Config():
    horizon = 14
    max_steps_dict = {
        "libero_spatial": 25,
        "libero_object": 25,
        "libero_goal": 25,
        "libero_10": 95
    }
    task_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] 
    num_episodes = 10
    
    EVAL_SEEDS = [42, 123, 2024, 3407, 10086]
    FIXED_STEPS = None

cfg = Config()
log = logging.getLogger(__name__)

########################################

# ========= Photos to list[list[list[int]]] =========
def encode_image_array(img_array: np.ndarray):
    return img_array.astype(np.uint8).tolist()

# ========= Quaternion to Axis-Angle =========
def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# ========= Observation to JSON-compatible dict =========
def obs_to_json_dict(obs, prompt, resize_size=448):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

    data = {
        "image": [
            encode_image_array(img),
            encode_image_array(wrist_img),
            encode_image_array(dummy_proc)
        ],
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }
    
    if cfg.FIXED_STEPS is not None:
        data["steps"] = cfg.FIXED_STEPS

    return data

# ========= Get the environment of LIBERO =========
def get_libero_env(task, resolution=448, seed=42):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

# ========= Save the video log =========
def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        # print(f"Video saved: {filepath} ({len(frames)} frames)")
    else:
        log.warning(f"⚠️ No frames to save. File not created: {filepath}")

# ========= Run Single Suite =========
# ========= Run Single Suite =========
async def run_suite(ws, task_suite_name: str, max_steps: int, num_episodes: int, horizon: int, seed: int, ckpt_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    total_success = 0
    total_episodes = 0
    total_steps = 0
    
    suite_vlm_latency = 0.0
    suite_act_latency = 0.0
    suite_tot_latency = 0.0
    suite_inf_count = 0
    suite_steps_used = []

    log.info(f"\n==================== Start task suite: {task_suite_name} (Seed: {seed}) ====================")

    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=448, seed=seed)

        task_success = 0
        task_episodes = min(num_episodes, len(initial_states))
        
        task_inference_latency_ms = 0.0
        task_inference_steps = 0
        stats_steps = []  
        stats_sims = []   
        stats_mags = []   

        for ep in range(task_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[ep])
            
            # Warmup
            for _ in range(10):
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    
            prompt = str(task_description)
            episode_done = False
            max_step = 0
            frames = []

            for step in range(max_steps):
                max_step += 1

                send_data = obs_to_json_dict(obs, prompt)
                await ws.send(json.dumps(send_data))

                result = await ws.recv()
                try:
                    resp_data = json.loads(result)
                    
                    if isinstance(resp_data, list):
                        actions = np.array(resp_data)
                        lat_tot, lat_vlm, lat_act, step_count, sim_val, mag_val = 0, 0, 0, 0, 0.0, 0.0
                    else:
                        actions = np.array(resp_data["action"])
                        lat_tot = resp_data.get("latency_total", 0.0)
                        lat_vlm = resp_data.get("latency_vlm", 0.0)
                        lat_act = resp_data.get("latency_action", 0.0)
                        step_count = resp_data.get("steps", 0)
                        sim_val = resp_data.get("sim", 0.0)
                        mag_val = resp_data.get("mag", 0.0)
                        
                        suite_tot_latency += lat_tot
                        suite_vlm_latency += lat_vlm
                        suite_act_latency += lat_act
                        suite_inf_count += 1
                        suite_steps_used.append(step_count)
                        
                        task_inference_latency_ms += lat_tot
                        task_inference_steps += 1
                        stats_steps.append(step_count)
                        stats_sims.append(sim_val)
                        stats_mags.append(mag_val)
                        
                except Exception as e:
                    log.error(f"❌ Action parsing failed: {e}")
                    break

                for i in range(horizon):
                    action = actions[i].tolist()
                    if action[6] > 0.5:
                        action[6] = -1
                    else:
                        action[6] = 1
                    
                    try:
                        obs, reward, done, info = env.step(action[:7])
                    except ValueError as ve:
                        log.error(f"❌ Invalid action: {ve}")
                        episode_done = False
                        break
                    
                    frame = np.hstack([
                        np.rot90(obs["agentview_image"], 2),
                        np.rot90(obs["robot0_eye_in_hand_image"], 2)
                    ])
                    frames.append(frame)

                    if done:
                        episode_done = True
                        task_success += 1
                        total_success += 1
                        total_steps += max_step
                        break
                        
                if episode_done:
                    break

            # save_video(frames, f"task{task_id+1}_episode{ep+1}.mp4", fps=30, save_dir=f"./video_log_file/{ckpt_name}/{task_suite_name}/seed_{seed}")

        total_episodes += task_episodes

        task_rate = task_success / max(1, task_episodes)
        avg_task_lat = task_inference_latency_ms / max(1, task_inference_steps)
        avg_diff_steps = sum(stats_steps) / len(stats_steps) if stats_steps else 0
        avg_sim = sum(stats_sims) / len(stats_sims) if stats_sims else 0
        min_sim = min(stats_sims) if stats_sims else 0
        avg_mag = sum(stats_mags) / len(stats_mags) if stats_mags else 0
        
        msg = (f"[Task {task_id} {task_suite_name}]->"
               f"success_rate={task_rate:.3f} "
               f"latency={avg_task_lat:.2f}ms "
               f"steps={avg_diff_steps:.1f}  sim_avg={avg_sim:.4f}  sim_min={min_sim:.4f}  mag_avg={avg_mag:.4f} "
               f"(s={task_success}, t={task_episodes}) "
               f"{task_description} finished {task_episodes} episodes")
        log.info(msg)

    return total_success, total_episodes, suite_vlm_latency, suite_act_latency, suite_tot_latency, suite_inf_count, suite_steps_used

# ========= Main Orchestrator =========
async def _amain(server_url: str, ckpt_name: str):
    log.info(f"==================================================")
    log.info(f"STARTING SEQUENTIAL EVALUATION ON LIBERO (Seeds: {cfg.EVAL_SEEDS})")
    log.info(f"==================================================")

    results = {"steps": [], "vlm": [], "action": [], "total": [], "freq": [], "sr": []}

    for i, seed in enumerate(cfg.EVAL_SEEDS):
        log.info(f"\n>>> [Progress {i+1}/{len(cfg.EVAL_SEEDS)}] RUNNING SEED: {seed} <<<\n")
        
        np.random.seed(seed)
        random.seed(seed)
        
        seed_success = 0
        seed_episodes = 0
        seed_vlm_lat = 0.0
        seed_act_lat = 0.0
        seed_tot_lat = 0.0
        seed_inf_count = 0
        seed_steps_used = []

        async with websockets.connect(server_url, max_size=100_000_000) as ws:
            for suite_name in cfg.task_suites:
                max_steps = cfg.max_steps_dict[suite_name]
                s_succ, s_eps, v_lat, a_lat, t_lat, i_cnt, s_used = await run_suite(
                    ws=ws, 
                    task_suite_name=suite_name,
                    max_steps=max_steps,
                    num_episodes=cfg.num_episodes,
                    horizon=cfg.horizon,
                    seed=seed,
                    ckpt_name=ckpt_name
                )
                seed_success += s_succ
                seed_episodes += s_eps
                seed_vlm_lat += v_lat
                seed_act_lat += a_lat
                seed_tot_lat += t_lat
                seed_inf_count += i_cnt
                seed_steps_used.extend(s_used)
        
        # Calculate seed metrics
        sr = (seed_success / max(1, seed_episodes)) * 100.0
        avg_vlm = seed_vlm_lat / max(1, seed_inf_count)
        avg_act = seed_act_lat / max(1, seed_inf_count)
        avg_tot = seed_tot_lat / max(1, seed_inf_count)
        avg_steps = sum(seed_steps_used) / max(1, len(seed_steps_used))
        freq = 1000.0 / avg_tot if avg_tot > 0 else 0.0
        
        results["steps"].append(avg_steps)
        results["vlm"].append(avg_vlm)
        results["action"].append(avg_act)
        results["total"].append(avg_tot)
        results["freq"].append(freq)
        results["sr"].append(sr)
        
        import gc
        gc.collect()

    def get_stats(key):
        arr = np.array(results[key])
        return np.mean(arr), np.std(arr)

    m_steps, s_steps = get_stats("steps")
    m_vlm, s_vlm = get_stats("vlm")
    m_act, s_act = get_stats("action")
    m_tot, s_tot = get_stats("total")
    m_freq, s_freq = get_stats("freq")
    m_sr, s_sr = get_stats("sr")

    log.info("\n" + "#"*80)
    log.info(f"             FINAL PAPER RESULTS (Avg over {len(cfg.EVAL_SEEDS)} Seeds)             ")
    log.info("#"*80)
    log.info(f"{'Metric':<25} | {'Mean':<10} | {'Std':<10}")
    log.info("-" * 60)
    log.info(f"{'Inference Steps':<25} | {m_steps:.2f}     | {s_steps:.2f}")
    log.info(f"{'Visual Encoder (ms)':<25} | {m_vlm:.2f}     | {s_vlm:.2f}")
    log.info(f"{'Action Head (ms)':<25} | {m_act:.2f}     | {s_act:.2f}")
    log.info(f"{'Total Latency (ms)':<25} | {m_tot:.2f}     | {s_tot:.2f}")
    log.info(f"{'Control Freq (Hz)':<25} | {m_freq:.2f}     | {s_freq:.2f}")
    log.info(f"{'Success Rate (%)':<25} | {m_sr:.2f}     | {s_sr:.2f}")
    log.info("#"*80)

    log.info("\n[LaTeX Table Row Generator]")
    log.info(f"% Copy this into your Table")
    setting_name = "AdaFlow" if cfg.FIXED_STEPS is None else f"Fixed-{cfg.FIXED_STEPS}"
    log.info(f"{setting_name} & {m_steps:.1f} & {m_vlm:.1f} & {m_act:.1f} $\\pm$ {s_act:.1f} & {m_tot:.1f} $\\pm$ {s_tot:.1f} & {m_freq:.1f} & {m_sr:.1f} $\\pm$ {s_sr:.1f} \\\\")
    log.info("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO Evo1 Client")
    parser.add_argument("--port", type=int, default=9011, help="Server Port")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint Dir (for logging name)")
    args = parser.parse_args()

    exp_name = os.path.basename(os.path.normpath(args.ckpt_dir))
    LOG_DIR = "log_file"
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = os.path.join(LOG_DIR, f"eval_sequential_libero_{exp_name}_{ts}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode='a'),
            logging.StreamHandler()
        ]
    )

    target_url = f"ws://127.0.0.1:{args.port}"
    asyncio.run(_amain(server_url=target_url, ckpt_name=exp_name))