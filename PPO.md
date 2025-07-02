TRPO 使用泰勒展开近似、共轭梯度、线性搜索等方法直接求解。PPO 的优化目标与 TRPO 相同，但 PPO 用了一些相对简单的方法来求解。并且大量的实验结果表明，与 TRPO 相比，PPO 能学习得一样好（甚至更快），这使得 PPO 成为非常流行的强化学习算法。

##  PPO-惩罚

PPO-惩罚（PPO-Penalty）用拉格朗日乘数法直接将 KL 散度的限制放进了目标函数中，这就变成了一个无约束的优化问题，在迭代的过程中不断更新 KL 散度前的系数。即：
$$
\arg\max_\theta \mathbb{E}_{s \sim \nu^{\pi_{\theta_k}}, a \sim \pi_{\theta_k}(\cdot|s)} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a) - \beta D_{KL}[\pi_{\theta_k}(\cdot|s), \pi_\theta(\cdot|s)] \right]
$$
令 $d_k = D_{KL}^{\pi_{\theta_k}}(\pi_k, \pi_\theta)$，$\beta$ 的更新规则如下：

1. 如果 $d_k < \delta / 1.5$，那么 $\beta_{k+1} = \beta_k / 2$
2. 如果 $d_k > \delta \times 1.5$，那么 $\beta_{k+1} = \beta_k \times 2$
3. 否则 $\beta_{k+1} = \beta_k$

其中，$\delta$ 是事先设定的一个超参数，用于限制学习策略和之前一轮策略的差距。

##  PPO-截断

PPO的另一种形式PPO-截断（PPO-Clip）更加直接，它在目标函数中进行限制，以保证新的参数和旧的参数的差距不会太大，即：
$$
\arg\max_\theta \mathbb{E}_{s \sim \nu^{\pi_{\theta_k}}, a \sim \pi_{\theta_k}(\cdot|s)} \left[ \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}A^{\pi_{\theta_k}}(s,a), \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}, 1-\epsilon, 1+\epsilon \right)A^{\pi_{\theta_k}}(s,a) \right) \right]
$$
其中 $\text{clip}(x, l, r) := \max(\min(x, r), l)$，即把 $x$ 限制在 $[l, r]$ 内。上式中 $\epsilon$ 是一个超参数，表示进行截断（clip）的范围。

如果 $A^{\pi_{\theta_k}}(s, a) > 0$，说明这个动作的价值高于平均，最大化这个式子会增大 $\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}$，但不会让其超过 $1+\epsilon$。反之，如果 $A^{\pi_{\theta_k}}(s, a) < 0$，最大化这个式子会减小 $\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}$，但不会让其超过 $1-\epsilon$。