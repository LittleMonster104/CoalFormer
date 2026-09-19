#!/usr/bin/env python3
"""
SMAC 3m - 补全所有Baseline方法
目标: 完整的6方法对比表格
- CoalFormer (已有)
- BRIDGE (已有)
- QMIX (已有)
- MAPPO (需要跑)
- VDN (需要跑)
- UneVEn (需要跑)
"""

import os, time, json, numpy as np, torch, torch.nn as nn
from collections import deque
import pandas as pd
from smac.env import StarCraft2Env

print("="*60)
print("SMAC 3m - 补全所有Baseline (MAPPO, VDN, UneVEn)")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ============================================================
# 实验配置
# ============================================================

EXPERIMENTS = []
maps = ['3m']
methods = ['VDN', 'MAPPO', 'UneVEn']  # 补充这3个
seeds = [0, 1, 2, 3, 4]

for map_name in maps:
    for method in methods:
        for seed in seeds:
            EXPERIMENTS.append({
                'map': map_name,
                'method': method,
                'seed': seed,
                'episodes': 1000,
                'id': f"{map_name}_{method}_seed{seed}"
            })

print(f"实验配置:")
print(f"  地图: {maps}")
print(f"  补充方法: {methods}")
print(f"  Seeds: {len(seeds)}个")
print(f"  总实验数: {len(EXPERIMENTS)}")
print(f"  预计时间: 4小时\n")

# ============================================================
# Agent实现
# ============================================================

class Agent(nn.Module):
    def __init__(self, obs_dim, n_agents, n_actions, method='VDN'):
        super().__init__()
        self.method = method
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        # 所有方法共用的encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        if method == 'MAPPO':
            # MAPPO: Policy + Value network
            self.policy = nn.Linear(64, n_actions)
            self.value = nn.Linear(64, 1)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        else:
            # VDN, UneVEn: Q-network
            self.q_net = nn.Linear(64, n_actions)
            
            if method == 'UneVEn':
                # UneVEn: 额外的coordination graph
                # 简化版本：用MLP学习agent之间的关系
                self.coord_net = nn.Sequential(
                    nn.Linear(64 * n_agents, 32),
                    nn.ReLU(),
                    nn.Linear(32, n_agents * n_agents)
                )
            
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        
        self.to(device)
    
    def forward(self, obs, agent_ids=None):
        is_batched = len(obs.shape) == 3
        
        if not is_batched:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        n_agents = obs.shape[1]
        
        obs_flat = obs.view(-1, obs.shape[-1])
        features = self.encoder(obs_flat)
        
        if self.method == 'MAPPO':
            # MAPPO: policy + value
            logits = self.policy(features)
            value = self.value(features)
            
            logits = logits.view(batch_size, n_agents, -1)
            value = value.view(batch_size, n_agents, -1)
            
            if not is_batched:
                logits = logits.squeeze(0)
                value = value.squeeze(0)
            
            return logits, value
        else:
            # VDN, UneVEn: Q-values
            q = self.q_net(features)
            q = q.view(batch_size, n_agents, -1)
            
            if not is_batched:
                q = q.squeeze(0)
            
            return q
    
    def get_actions(self, obs, avail_actions, agent_ids=None, eps=0.1):
        obs = obs.to(device)
        avail_actions = avail_actions.to(device)
        
        with torch.no_grad():
            if self.method == 'MAPPO':
                logits, _ = self.forward(obs, agent_ids)
                
                # Mask unavailable actions
                logits[avail_actions == 0] = -1e10
                
                # Sample from policy
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                
                return actions.cpu().numpy()
            else:
                q = self.forward(obs, agent_ids)
                
                q_masked = q.clone()
                q_masked[avail_actions == 0] = -1e10
                
                if np.random.random() < eps:
                    actions = []
                    for i in range(self.n_agents):
                        avail = np.where(avail_actions[i].cpu().numpy() == 1)[0]
                        actions.append(np.random.choice(avail))
                    return np.array(actions)
                else:
                    return q_masked.argmax(-1).cpu().numpy()
    
    def train_step(self, batch):
        obs, acts, rews, next_obs, dones, avail, next_avail, agent_ids = batch
        
        obs = obs.to(device)
        acts = acts.to(device)
        rews = rews.to(device)
        next_obs = next_obs.to(device)
        dones = dones.to(device)
        avail = avail.to(device)
        next_avail = next_avail.to(device)
        
        batch_size = obs.shape[0]
        
        if self.method == 'MAPPO':
            # MAPPO: PPO loss
            logits, values = self.forward(obs, agent_ids)
            
            # Policy loss (simplified, no old policy for now)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
            
            # Value loss
            target = rews + 0.99 * values.mean(1) * (1 - dones)
            value_loss = ((values.mean(1) - target) ** 2).mean()
            
            # Total loss (simplified)
            loss = -selected_log_probs.mean() + 0.5 * value_loss
        
        else:
            # VDN / UneVEn: Q-learning
            q = self.forward(obs, agent_ids)
            q_taken = q.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
            
            if self.method == 'VDN':
                # VDN: simple sum
                q_total = q_taken.sum(-1, keepdim=True)
            
            elif self.method == 'UneVEn':
                # UneVEn: weighted sum based on coordination graph
                features_flat = self.encoder(obs.view(-1, obs.shape[-1]))
                features_concat = features_flat.view(batch_size, -1)
                
                # Coordination weights
                coord_weights = self.coord_net(features_concat)
                coord_weights = coord_weights.view(batch_size, self.n_agents, self.n_agents)
                coord_weights = torch.softmax(coord_weights, dim=-1)
                
                # Weighted aggregation
                q_weighted = torch.bmm(coord_weights, q_taken.unsqueeze(-1)).squeeze(-1)
                q_total = q_weighted.sum(-1, keepdim=True)
            
            # Target
            with torch.no_grad():
                next_q = self.forward(next_obs, agent_ids)
                next_q[next_avail == 0] = -1e10
                next_q_max = next_q.max(-1)[0]
                
                if self.method == 'VDN':
                    target = rews + 0.99 * next_q_max.sum(-1, keepdim=True) * (1 - dones)
                elif self.method == 'UneVEn':
                    # Simplified: use sum for target
                    target = rews + 0.99 * next_q_max.sum(-1, keepdim=True) * (1 - dones)
            
            loss = ((q_total - target) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()

print("✅ 算法准备完成\n")

# ============================================================
# 运行实验
# ============================================================

results = []
start_all = time.time()

for i, exp in enumerate(EXPERIMENTS):
    print(f"\n[{i+1}/{len(EXPERIMENTS)}] {exp['id']}")
    start = time.time()
    
    np.random.seed(exp['seed'])
    torch.manual_seed(exp['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(exp['seed'])
    
    try:
        env = StarCraft2Env(map_name=exp['map'])
        info = env.get_env_info()
        
        agent = Agent(info['obs_shape'], info['n_agents'], info['n_actions'], exp['method'])
        
        rewards, wins = [], []
        buffer = deque(maxlen=2000)
        
        # Epsilon decay
        EXPLORE_RATIO = 0.3
        EXPLORE_EPS = int(exp['episodes'] * EXPLORE_RATIO)
        min_eps = 0.05
        eps = 1.0
        decay = (min_eps / eps) ** (1 / EXPLORE_EPS)
        
        for ep in range(exp['episodes']):
            env.reset()
            done, ep_rew = False, 0
            
            while not done:
                obs = np.array([env.get_obs_agent(j) for j in range(info['n_agents'])])
                avail = np.array([env.get_avail_agent_actions(j) for j in range(info['n_agents'])])
                agent_ids = torch.LongTensor(list(range(info['n_agents'])))
                
                obs_t = torch.FloatTensor(obs)
                avail_t = torch.FloatTensor(avail)
                
                # MAPPO不用epsilon（用policy sampling）
                if exp['method'] == 'MAPPO':
                    acts = agent.get_actions(obs_t, avail_t, agent_ids)
                else:
                    acts = agent.get_actions(obs_t, avail_t, agent_ids, eps)
                
                rew, done, info_step = env.step(acts.tolist())
                ep_rew += rew
                
                if not done:
                    next_obs = np.array([env.get_obs_agent(j) for j in range(info['n_agents'])])
                    next_avail = np.array([env.get_avail_agent_actions(j) for j in range(info['n_agents'])])
                else:
                    next_obs, next_avail = np.zeros_like(obs), np.zeros_like(avail)
                
                buffer.append((
                    torch.FloatTensor(obs),
                    torch.LongTensor(acts),
                    torch.FloatTensor([rew]),
                    torch.FloatTensor(next_obs),
                    torch.FloatTensor([float(done)]),
                    torch.FloatTensor(avail),
                    torch.FloatTensor(next_avail),
                    agent_ids
                ))
                
                if len(buffer) >= 32:
                    idx = np.random.choice(len(buffer), 32, replace=False)
                    batch = [buffer[j] for j in idx]
                    agent.train_step(tuple(torch.stack([b[k] for b in batch]) for k in range(8)))
            
            if exp['method'] != 'MAPPO':
                eps = max(min_eps, eps * decay)
            
            rewards.append(ep_rew)
            wins.append(float(info_step.get('battle_won', 0)))
            
            if (ep+1) % 100 == 0:
                print(f"  Ep{ep+1:4d}: R={np.mean(rewards[-100:]):6.2f}, W={np.mean(wins[-100:]):.1%}")
        
        env.close()
        elapsed = time.time() - start
        
        result = {
            'id': exp['id'], 
            'map': exp['map'], 
            'method': exp['method'], 
            'seed': exp['seed'],
            'final_reward': float(np.mean(rewards[-100:])),
            'final_win_rate': float(np.mean(wins[-100:])),
            'peak_win_rate': float(np.max(wins)),
            'time_min': elapsed/60
        }
        results.append(result)
        
        print(f"✅ Final W={result['final_win_rate']:.1%}, Peak={result['peak_win_rate']:.1%}, 用时={elapsed/60:.1f}min")
        
        with open('results_3m_complete_baselines.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append({'id': exp['id'], 'error': str(e)})

total_time = time.time() - start_all
print(f"\n{'='*60}")
print(f"实验完成! 总用时: {total_time/3600:.2f}小时")
print(f"{'='*60}")

with open('results_3m_complete_final.json', 'w') as f:
    json.dump(results, f, indent=2)

success = [r for r in results if 'error' not in r]
if success:
    df = pd.DataFrame(success)
    summary = df.groupby(['map', 'method']).agg({
        'final_win_rate': ['mean', 'std'],
        'peak_win_rate': 'max'
    }).round(3)
    print("\n实验汇总:")
    print(summary)
    df.to_csv('results_3m_complete_detailed.csv', index=False)

print("\n🎉 SMAC 3m完整Baseline补充完成！")
print("\n完整的6方法对比：")
print("  1. VDN (新)")
print("  2. MAPPO (新)")
print("  3. UneVEn (新)")
print("  4. QMIX (已有: 23.0%)")
print("  5. CoalFormer (已有: 34.4%)")
print("  6. BRIDGE (已有: 27.2% / 58.5%)")
