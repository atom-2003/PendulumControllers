import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


# 環境の作成
filepath = 'pendulum.xml'
filepath = os.path.abspath(filepath)
env = gym.make('InvertedPendulum-v5', xml_file=filepath)


# 学習パラメータの保存ディレクトリ
save_dir = "models/ppo_inverted_pendulum"
os.makedirs(save_dir, exist_ok=True)

# モデルの初期化
model = PPO("MlpPolicy", env, verbose=1)

# チェックポイントコールバック
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # 10,000ステップごとに保存
    save_path=save_dir,
    name_prefix="ppo_inverted_pendulum"
)

# 学習
print("Start training...")
model.learn(total_timesteps=100000, callback=checkpoint_callback)
print("Training finished.")

# モデルの最終保存
final_model_path = os.path.join(save_dir, "ppo_inverted_pendulum_final")
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

# 学習したモデルを使用したテスト
env = gym.make('InvertedPendulum-v5', xml_file=filepath, render_mode="human")

obs, _ = env.reset()

for _ in range(1000):  # テストエピソードの長さ
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
