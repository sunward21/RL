"""最小 CartPole 学习实验：随机对照 → PPO 训练 → 重载评估 → 动画。"""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


ENV_ID = "CartPole-v1"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUTPUT_DIR / "ppo_cartpole.zip"
TRAIN_SEED = 42
EVAL_SEED = 10_000  # 与训练种子分开，随机策略和 PPO 使用相同的 10 个初始状态。
EVAL_EPISODES = 10
REPLAY_SEED = 20_000  # 动画再用一个新初始状态；动画不等于统计评估。


def evaluate(model: PPO | None = None):
    """在新建的环境中跑 10 回合；model=None 表示随机策略。"""
    # 这里每次 gym.make 都创建一个独立环境，不复用训练时的小车状态。
    env = gym.make(ENV_ID)
    label = "随机策略" if model is None else "PPO（重新加载）"
    episodes = []
    try:
        for episode in range(EVAL_EPISODES):
            seed = EVAL_SEED + episode
            # reset 开始一个新回合，返回初始观测和辅助信息。
            # obs = [小车位置(m), 小车速度(m/s), 杆角度(rad), 杆角速度(rad/s)]。
            # 杆角度相对竖直方向；观测是 4 个数，不是动画中的图像。
            obs, info = env.reset(seed=seed)
            env.action_space.seed(seed)  # reset 的 seed 不负责动作空间的随机数。
            episode_return = 0.0
            steps = 0

            while True:
                if model is None:
                    # 动作只有两个：0 向左施力，1 向右施力；不是直接设定位置。
                    # 随机策略不看 obs，以相同概率选择左右。
                    action = env.action_space.sample()
                else:
                    # 推理：根据当前 obs 选动作，不调用 learn，不更新网络参数。
                    # deterministic=True 在离散动作中选概率最大的动作。
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action.item())

                # step 执行动作，并返回“下一时刻”的观测，而不是原来的观测。
                obs, reward, terminated, truncated, info = env.step(action)
                # 默认 CartPole 每步 +1，连失败的最后一步也 +1。
                # 因此一回合总奖励（return）等于走过的步数，最多 500。
                episode_return += float(reward)
                steps += 1

                # terminated：小车位置超出 ±2.4m，或杆角度超出约 ±12°。
                # truncated：达到外部的 500 步时间上限。
                # 两者都要求结束本回合，再 reset；它们可能同时为 True。
                # PPO 学习时要区分这两类结束，SB3 负责相关的价值估计处理。
                if terminated or truncated:
                    break

            record = {
                "episode": episode + 1,
                "seed": seed,
                "return": episode_return,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            episodes.append(record)
            print(
                f"{label} | 回合 {episode + 1:2d}/{EVAL_EPISODES} | "
                f"奖励 {episode_return:5.0f} | "
                f"terminated={terminated}, truncated={truncated}"
            )
    finally:
        env.close()

    returns = [record["return"] for record in episodes]
    result = {
        "policy": label,
        "env_id": ENV_ID,
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),  # 总体标准差，描述这 10 个结果的离散程度。
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = "random_eval.json" if model is None else "ppo_eval.json"
    (OUTPUT_DIR / filename).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{label}：平均奖励 {result['mean_return']:.1f} ± {result['std_return']:.1f}\n")
    return result


def train(total_timesteps):
    """训练阶段：让库负责收集交互经验和 PPO 参数更新。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Monitor 只记录每回合奖励和长度，不改变环境的奖励规则。
    train_env = Monitor(gym.make(ENV_ID), str(OUTPUT_DIR / "train.monitor.csv"))
    try:
        # MlpPolicy 是读取 4 个数的多层感知机；不需要图像卷积网络。
        # 训练时策略会采样动作探索，learn 用观测、动作和奖励更新参数。
        # 我们使用 SB3 的 PPO，不在这里实现 PPO 公式或优化器。
        model = PPO("MlpPolicy", train_env, device="cpu", seed=TRAIN_SEED, verbose=1)
        # timesteps 是环境交互步数，不是回合数，也不是参数更新次数。
        # 默认每批收集 2048 步，实际总步数会向上补齐到完整批次。
        model.learn(total_timesteps=total_timesteps)
        model.save(MODEL_PATH)
        metadata = {
            "env_id": ENV_ID,
            "seed": TRAIN_SEED,
            "device": str(model.device),
            "requested_timesteps": total_timesteps,
            "actual_timesteps": model.num_timesteps,
        }
        (OUTPUT_DIR / "training.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"已保存模型：{MODEL_PATH}")
    finally:
        train_env.close()
    # 故意不返回训练对象；后续步骤必须从磁盘加载模型。


def load_model():
    """重建模型并加载已学到的参数；加载本身不会继续训练。"""
    if not MODEL_PATH.is_file():
        raise SystemExit("还没有保存的模型，请先运行：bash run.sh train")
    return PPO.load(MODEL_PATH, device="cpu")


def replay(model: PPO, human=False):
    """用加载的模型跑一个完整回合：默认保存 GIF，也可打开实时窗口。"""
    if not human:
        # 离屏渲染无需桌面或显示器，GIF 也不依赖 ffmpeg。
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    env = gym.make(ENV_ID, render_mode="human" if human else "rgb_array")
    frames = []
    episode_return = 0.0
    try:
        obs, _ = env.reset(seed=REPLAY_SEED)
        if not human:
            frames.append(Image.fromarray(env.render()))
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            episode_return += float(reward)
            if not human:
                # 渲染只把状态画出来，图像不会送入本项目的策略网络。
                frames.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break
        if not human:
            OUTPUT_DIR.mkdir(exist_ok=True)
            gif_path = OUTPUT_DIR / "cartpole_ppo.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / env.metadata["render_fps"]),
                loop=0,
            )
            print(f"动画已保存：{gif_path}")
        print(f"回放奖励：{episode_return:.0f}，初始状态种子：{REPLAY_SEED}")
    finally:
        env.close()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("训练步数必须大于 0")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["random", "train", "evaluate", "replay", "all"])
    parser.add_argument("--timesteps", type=positive_int, default=50_000, help="训练交互步数，默认 50000")
    parser.add_argument("--human", action="store_true", help="回放时打开窗口，需本地图形桌面")
    args = parser.parse_args()
    # 小网络用一个 CPU 线程，避免大量线程调度开销；不影响系统线程设置。
    torch.set_num_threads(1)

    if args.command in ("random", "all"):
        evaluate()
    if args.command in ("train", "all"):
        train(args.timesteps)
    if args.command in ("evaluate", "all"):
        evaluate(load_model())
    if args.command in ("replay", "all"):
        replay(load_model(), human=args.human)


if __name__ == "__main__":
    main()
