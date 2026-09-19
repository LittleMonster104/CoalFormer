#!/usr/bin/env python3
"""
SMAC 3m - CoalFormer Prototype-based版本
使用论文中描述的方法：
1. Agent representation encoder
2. Learnable coalition prototypes  
3. Distance-based soft assignment
4. Two-phase training
"""

import os, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
import pandas as pd
from smac.env import StarCraft2Env

print("="*60)
print("SMAC 3m - CoalFormer Prototype-based (论文版本)")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ============================================================
# Prototype-based Coalition Module (论文方法)
# ============================================================

class PrototypeCoalitionModule(nn.Module):
    """论文中描述的Prototype-based Coalition Formation"""
    def __init__(self, feature_dim, n_coalitions):
        super().__init__()
        self.n_coalitions = n_coalitions
        
        # Agent representation encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Learnable coalition prototypes
        self.coalition_prototypes = nn.Parameter(
            torch.randn(n_coalitions, 32) * 0.1
        )
        
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, features):
        """
        Args:
            features: (batch_size, n_agents, feature_dim)
        Returns:
            coalition_probs: (batch_size, n_agents, n_coalitions)
            agent_repr: (batch_size, n_agents, 32)
        """
        batch_size, n_agents = features.shape[0], features.shape[1]
        
        # Encode agent representations
        agent_repr = self.agent_encoder(features)  # (B, N, 32)
        
        # Compute distances to prototypes
        agent_repr_expanded = agent_repr.unsqueeze(2)  # (B, N, 1, 32)
        prototypes_expanded = self.coalition_prototypes.unsqueeze(0).unsqueeze(0)  # (1, 1, K, 32)
        
        # Euclidean distance
        distances = torch.norm(
            agent_repr_expanded - prototypes_expanded,
            dim=-1
        )  # (B, N, K)
        
        # Soft assignment via softmax
        coalition_logits = -distances / (self.temperature.abs() + 1e-6)
        coalition_probs = F.softmax(coalition_logits, dim=-1)
        
        return coalition_probs, agent_repr

# ============================================================
# Agent with Prototype-based Coalition
# ============================================================

class Agent(nn.Module):
    def __init__(self, obs_dim, n_agents, n_actions):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_coalitions = 4
        
        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Q-network
        self.q_net = nn.Linear(64, n_actions)
        
        # Prototype-based coalition module
        self.coalition_module = PrototypeCoalitionModule(64, self.n_coalitions)
        
        # Mixer: coalition Q-values -> global Q
        self.mixer = nn.Sequential(
            nn.Linear(self.n_coalitions, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        self.to(device)
        
        self.training_phase = 1
        
        print(f"  → CoalFormer-Prototype: {self.n_coalitions} coalitions")
    
    def set_training_phase(self, phase):
        self.training_phase = phase
    
    def forward(self, obs, agent_ids=None):
        is_batched = len(obs.shape) == 3
        
        if not is_batched:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        n_agents = obs.shape[1]
        
        # Encode observations
        obs_flat = obs.view(-1, obs.shape[-1])
        features = self.encoder(obs_flat)
        features_3d = features.view(batch_size, n_agents, -1)
        
        # Get Q-values
        q = self.q_net(features)
        q = q.view(batch_size, n_agents, -1)
        
        # Get coalition assignments
        coalition_probs, agent_repr = self.coalition_module(features_3d)
        
        if not is_batched:
            q = q.squeeze(0)
            coalition_probs = coalition_probs.squeeze(0)
        
        return q, coalition_probs
    
    def get_actions(self, obs, avail_actions, agent_ids=None, eps=0.1):
        obs = obs.to(device)
        avail_actions = avail_actions.to(device)
        
        with torch.no_grad():
            q, _ = self.forward(obs, agent_ids)
            
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
        
        # Forward
        q, coalition_probs = self.forward(obs, agent_ids)
        q_taken = q.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        
        # Coalition-based value aggregation
        weighted_q = q_taken.unsqueeze(-1) * coalition_probs  # (B, N, K)
        q_by_coalition = weighted_q.sum(1)  # (B, K)
        q_total = self.mixer(q_by_coalition)  # (B, 1)
        
        # Regularization
        coalition_entropy = -(coalition_probs * torch.log(coalition_probs + 1e-8)).sum(-1).mean()
        
        prototypes = self.coalition_module.coalition_prototypes
        prototype_distances = torch.cdist(prototypes, prototypes, p=2)
        mask = (1 - torch.eye(self.n_coalitions, device=device))
        separation_loss = -torch.mean(prototype_distances * mask)
        
        # Target
        with torch.no_grad():
            next_q, next_coalition_probs = self.forward(next_obs, agent_ids)
            next_q[next_avail == 0] = -1e10
            next_q_max = next_q.max(-1)[0]
            
            weighted_next_q = next_q_max.unsqueeze(-1) * next_coalition_probs
            next_q_by_coalition = weighted_next_q.sum(1)
            target = rews + 0.99 * self.mixer(next_q_by_coalition) * (1 - dones)
        
        # Loss with phase-dependent regularization
        td_loss = ((q_total - target) ** 2).mean()
        
        if self.training_phase == 1:
            # Phase 1: Focus on Q-learning
            loss = td_loss + 0.001 * coalition_entropy + 0.0001 * separation_loss
        else:
            # Phase 2: Refine coalitions
            loss = td_loss + 0.01 * coalition_entropy + 0.001 * separation_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()

print("✅ Prototype-based CoalFormer准备完成\n")

# ============================================================
# 实验配置 - 只跑CoalFormer的5个seeds
# ============================================================

EXPERIMENTS = []
map_name = '3m'
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    EXPERIMENTS.append({
        'map': map_name,
        'seed': seed,
        'episodes': 1000,
        'id': f"3m_CoalFormer_Prototype_seed{seed}"
    })

print(f"实验配置:")
print(f"  地图: {map_name}")
print(f"  方法: CoalFormer (Prototype-based)")
print(f"  Seeds: {len(seeds)}个")
print(f"  预计时间: 1.5小时\n")

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
        
        agent = Agent(info['obs_shape'], info['n_agents'], info['n_actions'])
        
        rewards, wins = [], []
        buffer = deque(maxlen=2000)
        
        # Training schedule
        EXPLORE_RATIO = 0.3
        EXPLORE_EPS = int(exp['episodes'] * EXPLORE_RATIO)
        PHASE2_START = int(exp['episodes'] * 0.5)
        
        min_eps = 0.05
        eps = 1.0
        decay = (min_eps / eps) ** (1 / EXPLORE_EPS)
        
        print(f"  Exploration: {EXPLORE_EPS} eps, Phase2: {PHASE2_START} eps")
        
        for ep in range(exp['episodes']):
            if ep == PHASE2_START:
                agent.set_training_phase(2)
                print(f"  >>> 切换到Phase 2 (精细化coalition)")
            
            env.reset()
            done, ep_rew = False, 0
            
            while not done:
                obs = np.array([env.get_obs_agent(j) for j in range(info['n_agents'])])
                avail = np.array([env.get_avail_agent_actions(j) for j in range(info['n_agents'])])
                agent_ids = torch.LongTensor(list(range(info['n_agents'])))
                
                obs_t = torch.FloatTensor(obs)
                avail_t = torch.FloatTensor(avail)
                
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
            
            eps = max(min_eps, eps * decay)
            rewards.append(ep_rew)
            wins.append(float(info_step.get('battle_won', 0)))
            
            if (ep+1) % 100 == 0:
                print(f"  Ep{ep+1:4d}: R={np.mean(rewards[-100:]):6.2f}, W={np.mean(wins[-100:]):.1%}, ε={eps:.4f}")
        
        env.close()
        elapsed = time.time() - start
        
        result = {
            'id': exp['id'], 
            'map': exp['map'], 
            'method': 'CoalFormer-Prototype',
            'seed': exp['seed'],
            'final_reward': float(np.mean(rewards[-100:])),
            'final_win_rate': float(np.mean(wins[-100:])),
            'peak_win_rate': float(np.max(wins)),
            'time_min': elapsed/60
        }
        results.append(result)
        
        print(f"✅ Final W={result['final_win_rate']:.1%}, Peak={result['peak_win_rate']:.1%}, 用时={elapsed/60:.1f}min")
        
        with open('results_3m_coalformer_prototype.json', 'w') as f:
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

with open('results_3m_coalformer_prototype_final.json', 'w') as f:
    json.dump(results, f, indent=2)

if results:
    df = pd.DataFrame([r for r in results if 'error' not in r])
    print("\n实验汇总:")
    print(f"CoalFormer-Prototype (5 seeds):")
    print(f"  平均: {df['final_win_rate'].mean():.1%} ± {df['final_win_rate'].std():.1%}")
    print(f"  最高: {df['final_win_rate'].max():.1%}")
    print(f"  最低: {df['final_win_rate'].min():.1%}")
    print(f"  成功率: {(df['final_win_rate'] > 0.15).sum()}/5")

print("\n🎉 CoalFormer Prototype-based版本测试完成！")
print("这是论文中描述的正确方法。")
