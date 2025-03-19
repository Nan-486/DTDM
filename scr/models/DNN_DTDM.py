import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import torch.optim as optim


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(torch.device('cuda'))
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        x=x.to(torch.device('cuda'))
        emb=emb.to(torch.device('cuda'))
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# 强化学习环境类
class DiffusionRLEnv:
    def __init__(self,
                 user_item_matrix,
                 diffusion_model,
                 max_steps=10,
                 device="cuda"):
        """
        扩散过程的强化学习环境
        :param user_item_matrix: 用户-物品交互矩阵
        :param diffusion_model: 预训练的扩散模型
        :param max_steps: 最大去噪步数
        """
        self.device = device
        self.model = diffusion_model.to(device)
        user_item_matrix = torch.tensor(user_item_matrix.toarray(), dtype=torch.float32)
        self.user_item_matrix = user_item_matrix.to(device)
        self.max_steps = max_steps
        self.num_users, self.num_items = user_item_matrix.shape
        self.current_user = None
        self.current_step = 0
        self.noise_scale = 1.0  # 初始噪声强度

    def reset(self, user_id=None):
        """重置环境，开始新的生成过程"""
        self.current_step = 0
        self.current_user = user_id if user_id else torch.randint(0, self.num_users, (1,))

        # 初始化带噪声的推荐向量
        self.current_state = self.user_item_matrix[self.current_user] + \
                             torch.randn_like(self.user_item_matrix[self.current_user]) * self.noise_scale
        return self.current_state

    def step(self, action):
        """
        执行去噪动作
        :param action: 包含去噪参数的字典（用于控制模型行为）
        """
        # 使用扩散模型进行去噪
        with torch.no_grad():
            denoised = self.model(self.current_state, torch.tensor([self.current_step]))

        # 更新状态（添加探索噪声）
        self.current_state = action["guidance_scale"] * denoised + \
                             (1 - action["guidance_scale"]) * self.current_state + \
                             action["exploration_noise"]

        self.current_step += 1

        # 稀疏奖励设计：仅在最后一步计算推荐质量
        reward = 0.0
        done = self.current_step >= self.max_steps

        if done:
            # 最终奖励：推荐列表与用户真实偏好的相似度
            reward = self._calculate_final_reward(denoised)

        return self.current_state, reward, done, {}

    def _calculate_final_reward(self, denoised_vector):
        """计算最终推荐质量奖励"""
        true_pref = self.user_item_matrix[self.current_user]
        cosine_sim = torch.cosine_similarity(denoised_vector, true_pref, dim=-1)
        diversity = self._calculate_diversity(denoised_vector)
        return 0.7 * cosine_sim + 0.3 * diversity

    def _calculate_diversity(self, vector):
        """计算推荐多样性指标"""
        rec_items = torch.topk(vector, 10).indices
        item_features = ...  # 需要物品特征数据
        return torch.var(item_features[rec_items])  # 示例：特征方差


# PPO智能体类
class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dims=[256, 256],
                 lr=3e-4,
                 gamma=0.99,
                 clip_epsilon=0.2):
        # 策略网络（基于原始DNN修改）
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + 64, hidden_dims[0]),  # 64为时间嵌入维度
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], action_dim * 2)).to(torch.device('cuda'))  # 输出均值和方差

        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim + 64, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1)).to(torch.device('cuda'))

        self.optimizer = optim.Adam(list(self.policy_net.parameters()) +
                                    list(self.value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def get_action(self, state, timestep):
        """通过策略网络获取动作"""
        state=state.to(torch.device('cuda'))
        timestep=timestep.to(torch.device('cuda'))
        time_emb = timestep_embedding(timestep, 64).to(torch.device('cuda'))
        state_with_time = torch.cat([state, time_emb], dim=-1).to(torch.device('cuda'))

        # 获取动作分布参数
        mu_logstd = self.policy_net(state_with_time)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        std = torch.exp(log_std)

        # 创建正态分布
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return {
            "action": action,
            "log_prob": log_prob,
            "guidance_scale": torch.sigmoid(action[..., 0]),  # 示例参数
            "exploration_noise": action[..., 1:]  # 示例参数
        }

    def update(self, transitions):
        """PPO更新步骤"""
        states, actions, rewards, next_states, dones, log_probs_old = transitions
        states=states.to(torch.device('cuda'))
        actions=actions.to(torch.device('cuda'))
        rewards=rewards.to(torch.device('cuda'))
        log_probs_old=log_probs_old.to(torch.device('cuda'))

        # 计算折扣回报
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(torch.device('cuda'))

        # 计算价值估计
        time_embs = timestep_embedding(torch.arange(len(states)), 64).to(torch.device('cuda'))
        state_values = self.value_net(torch.cat([states, time_embs], dim=-1))

        # 计算优势
        advantages = returns - state_values.detach()

        # 计算新动作概率
        new_actions = self.get_action(states, torch.arange(len(states))).to(torch.device('cuda'))
        log_probs_new = new_actions["log_prob"]

        # PPO损失计算
        ratio = torch.exp(log_probs_new - log_probs_old)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 价值损失
        value_loss = F.mse_loss(state_values, returns)

        # 总损失
        total_loss = policy_loss + 0.5 * value_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()