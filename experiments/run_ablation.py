#!/usr/bin/env python3
"""
SMAC 3m - CoalFormer Ablation Studies
测试每个组件的贡献：
1. No Prototype (random coalition assignment)
2. No Coalition Aggregation (direct sum like VDN)
3. No Entropy Regularization
4. No Separation Loss
5. Single Phase Training (no curriculum)

只在4个成功的seeds上测试: 0, 2, 3, 4
"""

import os, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
import pandas as pd
from smac.env import StarCraft2Env

print("="*60)
print("SMAC 3m - CoalFormer Ablation Studies")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ============================================================
# Prototype-based Coalition Module
# ============================================================

class PrototypeCoalitionModule(nn.Module):
    def __init__(self, feature_dim, n_coalitions):
        super().__init__()
        self.n_coalitions = n_coalitions
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.coalition_prototypes = nn.Parameter(
            torch.randn(n_coalitions, 32) * 0.1
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, features):
        batch_size, n_agents = features.shape[0], features.shape[1]
        
        agent_repr = self.agent_encoder(features)
        
        agent_repr_expanded = agent_repr.unsqueeze(2)
        prototypes_expanded = self.coalition_prototypes.unsqueeze(0).unsqueeze(0)
        
        distances = torch.norm(
            agent_repr_expanded - prototypes_expanded,
            dim=-1
        )
        
        coalition_logits = -distances / (self.temperature.abs() + 1e-6)
        coalition_probs = F.softmax(coalition_logits, dim=-1)
        
        return coalition_probs, agent_repr

# ============================================================
# Agent with Ablation Options
# ============================================================

class Agent(nn.Module):
    def __init__(self, obs_dim, n_agents, n_actions, ablation='full'):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_coalitions = 4
        self.ablation = ablation
        
        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Q-network
        self.q_net = nn.Linear(64, n_actions)
        
        # Coalition module (different based on ablation)
        if ablation != 'no_coalition_agg':
            self.coalition_module = PrototypeCoalitionModule(64, self.n_coalitions)
            self.mixer = nn.Sequential(
                nn.Linear(self.n_coalitions, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        self.to(device)
        
        self.training_phase = 1
        
        ablation_names = {
            'full': 'Full CoalFormer',
            'no_prototype': 'Random Coalition',
            'no_coalition_agg': 'No Coalition (VDN-like)',
            'no_entropy': 'No Entropy Reg',
            'no_separation': 'No Separation Loss',
            'single_phase': 'Single Phase Training'
        }
        print(f"  → {ablation_names[ablation]}")
    
    def set_training_phase(self, phase):
        self.training_phase = phase
    
    def forward(self, obs, agent_ids=None):
        is_batched = len(obs.shape) == 3
        
        if not is_batched:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        n_agents = obs.shape[1]
        
        obs_flat = obs.view(-1, obs.shape[-1])
        features = self.encoder(obs_flat)
        features_3d = features.view(batch_size, n_agents, -1)
        
        q = self.q_net(features)
        q = q.view(batch_size, n_agents, -1)
        
        if self.ablation == 'no_coalition_agg':
            # Like VDN: no coalition
            if not is_batched:
                q = q.squeeze(0)
            return q, None
        else:
            # Get coalition assignments
            if self.ablation == 'no_prototype':
                # Random assignment instead of prototype-based
                coalition_probs = torch.ones(batch_size, n_agents, self.n_coalitions, device=device)
                coalition_probs = coalition_probs / self.n_coalitions
                agent_repr = None
            else:
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
        
        batch_size = obs.shape[0]
        
        # Forward
        q, coalition_probs = self.forward(obs, agent_ids)
        q_taken = q.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        
        # Compute Q_total based on ablation
        if self.ablation == 'no_coalition_agg':
            # VDN-like: direct sum
            q_total = q_taken.sum(-1, keepdim=True)
        else:
            # Coalition-based aggregation
            weighted_q = q_taken.unsqueeze(-1) * coalition_probs
            q_by_coalition = weighted_q.sum(1)
            q_total = self.mixer(q_by_coalition)
        
        # Regularization (depends on ablation)
        reg_loss = 0.0
        
        if self.ablation not in ['no_coalition_agg', 'no_prototype']:
            # Entropy regularization
            if self.ablation != 'no_entropy':
                coalition_entropy = -(coalition_probs * torch.log(coalition_probs + 1e-8)).sum(-1).mean()
            else:
                coalition_entropy = 0.0
            
            # Separation loss
            if self.ablation != 'no_separation':
                prototypes = self.coalition_module.coalition_prototypes
                prototype_distances = torch.cdist(prototypes, prototypes, p=2)
                mask = (1 - torch.eye(self.n_coalitions, device=device))
                separation_loss = -torch.mean(prototype_distances * mask)
            else:
                separation_loss = 0.0
            
            # Phase-dependent weights
            if self.ablation == 'single_phase':
                # Fixed weights throughout
                lambda_ent = 0.005
                lambda_sep = 0.0005
            else:
                # Two-phase curriculum
                if self.training_phase == 1:
                    lambda_ent = 0.001
                    lambda_sep = 0.0001
                else:
                    lambda_ent = 0.01
                    lambda_sep = 0.001
            
            reg_loss = lambda_ent * coalition_entropy + lambda_sep * separation_loss
        
        # Target
        with torch.no_grad():
            next_q, next_coalition_probs = self.forward(next_obs, agent_ids)
            next_q[next_avail == 0] = -1e10
            next_q_max = next_q.max(-1)[0]
            
            if self.ablation == 'no_coalition_agg':
                target = rews + 0.99 * next_q_max.sum(-1, keepdim=True) * (1 - dones)
            else:
                weighted_next_q = next_q_max.unsqueeze(-1) * next_coalition_probs
                next_q_by_coalition = weighted_next_q.sum(1)
                target = rews + 0.99 * self.mixer(next_q_by_coalition) * (1 - dones)
        
        # Total loss
        td_loss = ((q_total - target) ** 2).mean()
        loss = td_loss + reg_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()

print("✅ Ablation Agent准备完成\n")

# ============================================================
# 实验配置
# ============================================================

EXPERIMENTS = []
map_name = '3m'
# 只测试4个成功的seeds
successful_seeds = [0, 2, 3, 4]

ablations = [
    'full',
    'no_prototype',
    'no_coalition_agg',
    'no_entropy',
    'no_separation',
    'single_phase'
]

for ablation in ablations:
    for seed in successful_seeds:
        EXPERIMENTS.append({
            'map': map_name,
            'ablation': ablation,
            'seed': seed,
            'episodes': 1000,
            'id': f"3m_{ablation}_seed{seed}"
        })

print(f"实验配置:")
print(f"  地图: {map_name}")
print(f"  Ablations: {len(ablations)}个")
print(f"  Seeds: {len(successful_seeds)}个 (只测成功的)")
print(f"  总实验数: {len(EXPERIMENTS)}")
print(f"  预计时间: 4小时\n")

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
        
        agent = Agent(info['obs_shape'], info['n_agents'], info['n_actions'], exp['ablation'])
        
        rewards, wins = [], []
        buffer = deque(maxlen=2000)
        
        EXPLORE_RATIO = 0.3
        EXPLORE_EPS = int(exp['episodes'] * EXPLORE_RATIO)
        PHASE2_START = int(exp['episodes'] * 0.5)
        
        min_eps = 0.05
        eps = 1.0
        decay = (min_eps / eps) ** (1 / EXPLORE_EPS)
        
        for ep in range(exp['episodes']):
            if ep == PHASE2_START and exp['ablation'] != 'single_phase':
                agent.set_training_phase(2)
            
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
            
            if (ep+1) % 200 == 0:
                print(f"  Ep{ep+1:4d}: W={np.mean(wins[-100:]):.1%}")
        
        env.close()
        elapsed = time.time() - start
        
        result = {
            'id': exp['id'], 
            'map': exp['map'],
            'ablation': exp['ablation'],
            'seed': exp['seed'],
            'final_reward': float(np.mean(rewards[-100:])),
            'final_win_rate': float(np.mean(wins[-100:])),
            'peak_win_rate': float(np.max(wins)),
            'time_min': elapsed/60
        }
        results.append(result)
        
        print(f"✅ Final W={result['final_win_rate']:.1%}, 用时={elapsed/60:.1f}min")
        
        with open('results_3m_ablation.json', 'w') as f:
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

with open('results_3m_ablation_final.json', 'w') as f:
    json.dump(results, f, indent=2)

success = [r for r in results if 'error' not in r]
if success:
    df = pd.DataFrame(success)
    
    print("\n📊 Ablation汇总:")
    summary = df.groupby('ablation')['final_win_rate'].agg(['mean', 'std']).round(3)
    summary['delta'] = summary['mean'] - df[df['ablation']=='full']['final_win_rate'].mean()
    summary = summary.sort_values('mean', ascending=False)
    print(summary)
    
    df.to_csv('results_3m_ablation_detailed.csv', index=False)

print("\n🎉 Ablation Studies完成！")
