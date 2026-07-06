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
    
    # EVAL_SEEDS = [42, 123, 2024, 3407, 10086]
    EVAL_SEEDS = [42]

cfg = Config()
log = logging.getLogger(__name__)
SOLVER_NAME = "probeflow"
FIXED_STEPS = None
VIDEO_SAVE_DIR = "videos_libero_eval"
LOG_PATH = ""

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

    request_steps = FIXED_STEPS
    if SOLVER_NAME in {"adaflow", "probeflow"} and FIXED_STEPS is None:
        request_steps = None

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
        "solver": SOLVER_NAME,
        "steps": request_steps,
    }
    

    return data

# ========= Get the environment of LIBERO =========
def get_libero_env(task, resolution=448, seed=42):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def project_3d_to_2d_libero(env, xyz, camera_name="agentview", img_size=448):
    import numpy as np
    
    sim = env.env.sim if hasattr(env, 'env') else env.sim
    model = sim.model
    data = sim.data
    
    cam_id = model.camera_name2id(camera_name)
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * img_size / np.tan(fovy * np.pi / 360)
    
    pos = data.cam_xpos[cam_id]
    xmat = data.cam_xmat[cam_id].reshape(3, 3)
    
    # MuJoCo to OpenCV coordinate transformation
    R_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ xmat.T
    t_cv = -R_cv @ pos
    
    # 投影到相机平面
    p_cam = R_cv @ np.array(xyz) + t_cv
    
    # 获取原始图像上的像素坐标 (Top-Left is 0,0)
    u = f * p_cam[0] / p_cam[2] + img_size / 2.0
    v = f * p_cam[1] / p_cam[2] + img_size / 2.0
    
    # 核心修复点：因为你的 background_img 做了 [::-1, ::-1] 的180度翻转
    # 物理坐标 u 和 v 必须做完全相同的翻转才能对齐！
    u = img_size - 1 - u
    v = img_size - 1 - v
    
    return u, v

def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        # print(f"Video saved: {filepath} ({len(frames)} frames)")
    else:
        log.warning(f"⚠️ No frames to save. File not created: {filepath}")


def get_setting_display_name() -> str:
    if SOLVER_NAME == "adaflow":
        return "AdaFlow (Adaptive)"
    if SOLVER_NAME == "probeflow":
        return "ProbeFlow (Adaptive)"
    if SOLVER_NAME == "dpm_multistep":
        return f"DPM-Multistep-{FIXED_STEPS}" if FIXED_STEPS is not None else "DPM-Multistep"
    if SOLVER_NAME == "rk45":
        return f"RK45-{FIXED_STEPS}" if FIXED_STEPS is not None else "RK45"
    if SOLVER_NAME == "heun":
        return f"Heun-{FIXED_STEPS}" if FIXED_STEPS is not None else "Heun"
    if SOLVER_NAME == "euler":
        return f"Euler-{FIXED_STEPS}" if FIXED_STEPS is not None else "Euler"
    if FIXED_STEPS is not None:
        return f"{str(SOLVER_NAME).upper()}-{FIXED_STEPS}"
    return str(SOLVER_NAME)

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

            trajectory_data = [] 
            background_img = None

            for step in range(max_steps):
                max_step += 1

                if background_img is None:
                    background_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

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
                    log.error(f"Action parsing failed: {e}")
                    break

                TARGET_FPS = 10.0  
                MS_PER_FRAME = 1000.0 / TARGET_FPS  
                
                demo_frame = np.hstack([
                    np.rot90(obs["agentview_image"], 2),
                    np.rot90(obs["robot0_eye_in_hand_image"], 2)
                ])
                demo_frame = np.ascontiguousarray(demo_frame)
                
                is_baseline = send_data.get("steps", None) is not None
                method_name = get_setting_display_name()
                
                # 注意：LIBERO 这里的 imageio 保存的是 RGB 格式，所以颜色按照 (R, G, B) 配置
                theme_color = (255, 0, 0) if is_baseline else (30, 144, 255) # Baseline红色，Ours科技蓝
                text_green = (0, 255, 0)

                def draw_text_with_bg(img, text, pos, font_scale, color, thickness):
                    import cv2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    x, y = pos
                    cv2.rectangle(img, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1) 
                    cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

                # 绘制大脑思考期间的数据面板 (HUD)
                draw_text_with_bg(demo_frame, method_name, (15, 35), 0.8, theme_color, 2)
                draw_text_with_bg(demo_frame, f"ODE Steps: {step_count}", (15, 75), 0.7, text_green, 2)
                draw_text_with_bg(demo_frame, f"Latency: {lat_tot:.1f} ms", (15, 110), 0.7, text_green, 2)

                freeze_frames = max(1, int(round(lat_tot / MS_PER_FRAME)))

                for _ in range(freeze_frames):
                    frames.append(demo_frame.copy())

                segment_uvs = [] 
                u, v = project_3d_to_2d_libero(env, obs["robot0_eef_pos"], img_size=448)
                segment_uvs.append([u, v])

                for i in range(horizon):
                    action = actions[i].tolist()
                    if action[6] > 0.5:
                        action[6] = -1
                    else:
                        action[6] = 1
                    
                    try:
                        obs, reward, done, info = env.step(action[:7])
                    except ValueError as ve:
                        episode_done = False
                        break
                    
                    current_xyz = obs["robot0_eef_pos"]
                    u, v = project_3d_to_2d_libero(env, current_xyz, img_size=448)
                    segment_uvs.append([u, v])

                    # 提取动作执行期间的画面
                    exec_frame = np.hstack([
                        np.rot90(obs["agentview_image"], 2),
                        np.rot90(obs["robot0_eye_in_hand_image"], 2)
                    ])
                    exec_frame = np.ascontiguousarray(exec_frame)
                    
                    # 动作执行时，屏幕上的 Latency 和 Steps 保持上一次推理的值不变
                    draw_text_with_bg(exec_frame, method_name, (15, 35), 0.8, theme_color, 2)
                    draw_text_with_bg(exec_frame, f"ODE Steps: {step_count}", (15, 75), 0.7, text_green, 2)
                    draw_text_with_bg(exec_frame, f"Latency: {lat_tot:.1f} ms", (15, 110), 0.7, text_green, 2)
                    
                    frames.append(exec_frame)

                    if done:
                        episode_done = True
                        task_success += 1
                        total_success += 1
                        total_steps += max_step
                        break
                
                if len(segment_uvs) > 0:
                    trajectory_data.append({
                        'uv_path': segment_uvs,
                        'n_steps': step_count
                     })

                if episode_done:
                    break
            
            if len(frames) > 0:
                video_name = f"task{task_id:02d}_{task_suite_name}_ep{ep:02d}.mp4"
                save_video(frames, filename=video_name, fps=10, save_dir=VIDEO_SAVE_DIR)

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
    setting_name = get_setting_display_name()
    log.info(f"{setting_name} & {m_steps:.1f} & {m_vlm:.1f} & {m_act:.1f} $\\pm$ {s_act:.1f} & {m_tot:.1f} $\\pm$ {s_tot:.1f} & {m_freq:.1f} & {m_sr:.1f} $\\pm$ {s_sr:.1f} \\\\")
    log.info("="*80 + "\n")

if __name__ == "__main__":
    def parse_seeds(seed_text: str):
        return [int(x.strip()) for x in seed_text.split(",") if x.strip()]

    parser = argparse.ArgumentParser(description="LIBERO Evo1 Client")
    parser.add_argument("--port", type=int, default=9011, help="Server Port")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint Dir (for logging name)")
    parser.add_argument("--exp_name", type=str, default=None, help="Explicit experiment name used in log filename")
    parser.add_argument("--solver", type=str, default="adaflow", choices=["adaflow", "probeflow", "euler", "rk45", "dpm_multistep", "heun"], help="Action solver")
    parser.add_argument("--steps", type=int, default=None, help="Fixed inference steps. Leave unset for adaptive solvers.")
    parser.add_argument("--seeds", type=str, default="42,123,2024,3407,10086", help="Comma-separated eval seeds")
    parser.add_argument("--episodes", type=int, default=cfg.num_episodes, help="Episodes per task")
    parser.add_argument("--save_video", action="store_true", help="Save rollout videos")
    args = parser.parse_args()

    if args.solver in {"euler", "rk45", "dpm_multistep", "heun"} and args.steps is None:
        raise ValueError("--steps is required for fixed-step solvers")

    SOLVER_NAME = args.solver
    FIXED_STEPS = args.steps
    cfg.EVAL_SEEDS = parse_seeds(args.seeds)
    cfg.num_episodes = args.episodes

    exp_name = args.exp_name if args.exp_name else os.path.basename(os.path.normpath(args.ckpt_dir))
    LOG_DIR = "log_file"
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = os.path.join(LOG_DIR, f"eval_sequential_libero_{exp_name}_{ts}.txt")
    VIDEO_SAVE_DIR = f"videos_{exp_name}" if args.save_video else "videos_libero_eval"
    
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
