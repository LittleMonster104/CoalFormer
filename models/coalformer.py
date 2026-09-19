"""
CoalFormer Model - Prototype-based Coalition Formation

Official implementation matching the ICLR 2027 paper:
"CoalFormer: Learning Emergent Coalition Structures via 
Differentiable Prototype Clustering for Multi-Agent Coordination"

This module implements the core CoalFormer architecture with:
1. Learnable coalition prototypes
2. Distance-based soft assignment
3. Coalition-aware value decomposition
4. Two-phase regularization training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PrototypeCoalitionModule(nn.Module):
    """
    Prototype-based Coalition Formation Module
    
    Implements the differentiable prototype clustering described in the paper.
    
    Args:
        feature_dim (int): Dimension of agent features
        n_coalitions (int): Number of coalition prototypes (K in paper)
        coalition_dim (int): Dimension of coalition representation space
    """
    def __init__(self, feature_dim, n_coalitions, coalition_dim=32):
        super().__init__()
        self.n_coalitions = n_coalitions
        self.coalition_dim = coalition_dim
        
        # Agent representation encoder
        # Maps agent features to coalition assignment space
        self.agent_encoder = nn.Sequential(
            nn.Linear(feature_dim, coalition_dim),
            nn.ReLU(),
            nn.Linear(coalition_dim, coalition_dim)
        )
        
        # Learnable coalition prototypes {p_k}
        # Initialized with small random values
        self.coalition_prototypes = nn.Parameter(
            torch.randn(n_coalitions, coalition_dim) * 0.1
        )
        
        # Temperature parameter τ (learnable)
        # Controls sharpness of soft assignment
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, features):
        """
        Compute soft coalition assignments via prototype clustering
        
        Args:
            features: (batch_size, n_agents, feature_dim)
            
        Returns:
            coalition_probs: (batch_size, n_agents, n_coalitions)
                Soft assignment probabilities π_{ik}
            agent_repr: (batch_size, n_agents, coalition_dim)
                Agent representations in coalition space z_i
        """
        batch_size, n_agents, _ = features.shape
        
        # Encode agents to coalition space
        agent_repr = self.agent_encoder(features)  # (B, N, d')
        
        # Expand dimensions for broadcasting
        agent_repr_expanded = agent_repr.unsqueeze(2)  # (B, N, 1, d')
        prototypes_expanded = self.coalition_prototypes.unsqueeze(0).unsqueeze(0)  # (1, 1, K, d')
        
        # Compute Euclidean distances ||z_i - p_k||
        distances = torch.norm(
            agent_repr_expanded - prototypes_expanded,
            dim=-1
        )  # (B, N, K)
        
        # Temperature-scaled negative distances
        # s_{ik} = -||z_i - p_k||^2 / τ
        coalition_logits = -distances / (self.temperature.abs() + 1e-6)
        
        # Softmax for soft assignment
        # π_{ik} = exp(s_{ik}) / Σ_j exp(s_{ij})
        coalition_probs = F.softmax(coalition_logits, dim=-1)
        
        return coalition_probs, agent_repr


class CoalFormerAgent(nn.Module):
    """
    Complete CoalFormer Agent with Prototype Clustering
    
    Implements the full architecture:
    1. Observation encoding
    2. Individual Q-networks
    3. Prototype-based coalition formation
    4. Coalition Q-value aggregation
    5. Monotonic mixing (simplified)
    
    Args:
        obs_dim (int): Observation dimension
        n_agents (int): Number of agents
        n_actions (int): Number of actions per agent
        n_coalitions (int): Number of coalitions (K=4 in paper)
        hidden_dim (int): Hidden layer dimension
    """
    def __init__(self, obs_dim, n_agents, n_actions, n_coalitions=4, hidden_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_coalitions = n_coalitions
        self.hidden_dim = hidden_dim
        
        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Individual Q-networks (one per agent)
        self.q_net = nn.Linear(hidden_dim, n_actions)
        
        # Prototype-based coalition module
        self.coalition_module = PrototypeCoalitionModule(
            feature_dim=hidden_dim,
            n_coalitions=n_coalitions
        )
        
        # Mixing network: coalition Q-values -> global Q
        # Note: For strict IGM, should use monotonic weights
        self.mixer = nn.Sequential(
            nn.Linear(n_coalitions, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Training phase for two-phase regularization
        self.training_phase = 1
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        
    def forward(self, obs, agent_ids=None):
        """
        Forward pass
        
        Args:
            obs: (batch_size, n_agents, obs_dim) or (n_agents, obs_dim)
            agent_ids: Optional agent IDs (not used in homogeneous case)
            
        Returns:
            q: (batch_size, n_agents, n_actions) - Individual Q-values
            coalition_probs: (batch_size, n_agents, n_coalitions) - Soft assignments
        """
        # Handle both batched and unbatched inputs
        is_batched = len(obs.shape) == 3
        if not is_batched:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        n_agents = obs.shape[1]
        
        # Encode observations
        obs_flat = obs.view(-1, obs.shape[-1])
        features = self.encoder(obs_flat)
        features_3d = features.view(batch_size, n_agents, -1)
        
        # Get individual Q-values
        q = self.q_net(features)
        q = q.view(batch_size, n_agents, -1)
        
        # Get coalition assignments
        coalition_probs, agent_repr = self.coalition_module(features_3d)
        
        if not is_batched:
            q = q.squeeze(0)
            coalition_probs = coalition_probs.squeeze(0)
        
        return q, coalition_probs
    
    def compute_coalition_q(self, q_values, coalition_probs, actions):
        """
        Compute coalition Q-values
        
        Q_coal^k = Σ_i π_{ik} · Q_i(o_i, a_i)
        
        Args:
            q_values: (batch_size, n_agents, n_actions)
            coalition_probs: (batch_size, n_agents, n_coalitions)
            actions: (batch_size, n_agents)
            
        Returns:
            coalition_q: (batch_size, n_coalitions)
        """
        batch_size = q_values.shape[0]
        
        # Select Q-values for chosen actions
        q_chosen = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Aggregate by coalition
        coalition_q = torch.zeros(
            batch_size, self.n_coalitions,
            device=q_values.device
        )
        
        for k in range(self.n_coalitions):
            coalition_q[:, k] = (coalition_probs[:, :, k] * q_chosen).sum(dim=1)
        
        return coalition_q
    
    def compute_regularization(self, coalition_probs):
        """
        Compute regularization losses
        
        Args:
            coalition_probs: (batch_size, n_agents, n_coalitions)
            
        Returns:
            entropy_loss: Encourages diverse assignments
            separation_loss: Encourages prototype separation
        """
        # Entropy regularization
        # L_ent = -Σ_i Σ_k π_{ik} log π_{ik}
        coalition_entropy = -(coalition_probs * 
                             torch.log(coalition_probs + 1e-8)).sum(-1).mean()
        
        # Prototype separation loss
        # L_sep = -Σ_{k≠j} ||p_k - p_j||^2
        prototypes = self.coalition_module.coalition_prototypes
        n_proto = prototypes.shape[0]
        
        separation_loss = 0
        for i in range(n_proto):
            for j in range(i + 1, n_proto):
                separation_loss -= torch.norm(prototypes[i] - prototypes[j]) ** 2
        
        separation_loss = separation_loss / (n_proto * (n_proto - 1) / 2)
        
        return coalition_entropy, separation_loss
    
    def train_step(self, batch):
        """
        Single training step with two-phase regularization
        
        Args:
            batch: Tuple of (obs, actions, rewards, next_obs, dones, 
                            avail_actions, next_avail_actions, agent_ids)
            
        Returns:
            loss: Scalar loss value
        """
        obs, acts, rews, next_obs, dones, avail, next_avail, agent_ids = batch
        
        # Forward pass
        q, coalition_probs = self.forward(obs, agent_ids)
        
        # Compute coalition Q-values
        coalition_q = self.compute_coalition_q(q, coalition_probs, acts)
        
        # Mix to get total Q
        q_total = self.mixer(coalition_q).squeeze(-1)
        
        # Compute target
        with torch.no_grad():
            next_q, next_coalition_probs = self.forward(next_obs, agent_ids)
            next_q_max = next_q.max(-1)[0]
            
            weighted_next_q = next_q_max.unsqueeze(-1) * next_coalition_probs
            next_q_by_coalition = weighted_next_q.sum(1)
            target = rews + 0.99 * self.mixer(next_q_by_coalition).squeeze(-1) * (1 - dones)
        
        # TD loss
        td_loss = ((q_total - target) ** 2).mean()
        
        # Regularization
        coalition_entropy, separation_loss = self.compute_regularization(coalition_probs)
        
        # Two-phase regularization schedule
        if self.training_phase == 1:
            # Phase 1 (episodes 1-500): Weak regularization
            loss = td_loss + 0.001 * coalition_entropy + 0.0001 * separation_loss
        else:
            # Phase 2 (episodes 501-1000): Strong regularization
            loss = td_loss + 0.01 * coalition_entropy + 0.001 * separation_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def set_training_phase(self, phase):
        """Set training phase for two-phase regularization"""
        self.training_phase = phase
    
    def get_actions(self, obs, avail_actions, agent_ids=None, eps=0.1):
        """
        Select actions using epsilon-greedy policy
        
        Args:
            obs: (n_agents, obs_dim)
            avail_actions: (n_agents, n_actions)
            agent_ids: Optional agent IDs
            eps: Exploration rate
            
        Returns:
            actions: (n_agents,) numpy array
        """
        with torch.no_grad():
            q, _ = self.forward(obs, agent_ids)
            
            # Mask unavailable actions
            q_masked = q.clone()
            q_masked[avail_actions == 0] = -1e10
            
            # Epsilon-greedy
            if np.random.random() < eps:
                actions = []
                for i in range(self.n_agents):
                    avail = np.where(avail_actions[i].cpu().numpy() == 1)[0]
                    actions.append(np.random.choice(avail))
                return np.array(actions)
            else:
                return q_masked.argmax(-1).cpu().numpy()


# Export main classes
__all__ = ['PrototypeCoalitionModule', 'CoalFormerAgent']
