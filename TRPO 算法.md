基于策略的方法包括策略梯度算法和 Actor-Critic 算法。这些方法虽然简单、直观，但在实际应用过程中会遇到训练不稳定的情况。

当策略网络是深度模型时，沿着策略梯度更新参数，很有可能由于步长太长，策略突然显著变差，进而影响训练效果。（和监督学习不同，**监督学习的训练数据分布是始终不变的，而在强化学习中，因为策略的改变会导致环境的改变，进而会导致采样的数据的分布也发生改变**，在深度网络中，单次更新的步长可能过大，导致新策略与旧策略产生巨大差异，从而进入一个“采集坏数据 -> 得到坏梯度 -> 策略更坏”的恶性循环中）

我们考虑在更新时找到一块**信任区域**（trust region），在这个区域上更新策略时能够得到某种策略性能的安全性保证，这就是**信任区域策略优化**（trust region policy optimization，TRPO）算法的主要思想。

假设当前策略为 $\pi_\theta$，参数为 $\theta$。我们考虑如何借助当前的数据找到一个更优的参数 $\theta'$，使得 $J(\theta') \ge J(\theta)$。具体来说，由于初始状态 $s_0$ 的分布和策略无关，因此上述优化目标 $J(\theta)$ 可以写成新策略 $\pi_{\theta'}$ 的期望形式： $$ \begin{aligned} J(\theta') &= \mathbb{E}_{s_0}[V^{\pi_{\theta'}}(s_0)] \\ &= \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t V^{\pi_\theta}(s_t) - \sum_{t=1}^\infty \gamma^t V^{\pi_\theta}(s_t) \right] \\ &= -\mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t (V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)) \right] \end{aligned} $$ 基于以上等式，我们可以推导新旧策略的目标函数之间的差距： $$ \begin{aligned} J(\theta') - J(\theta) &= \mathbb{E}_{s_0}[V^{\pi_{\theta'}}(s_0)] - \mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)] \\ &= \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t (r(s_t, a_t) + \mathbb{E}_{\pi_{\theta'}}[V^{\pi_\theta}(s_{t+1})] - V^{\pi_\theta}(s_t)) \right] \\ &= \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t (r(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)) \right] \end{aligned} $$ 将时序差分残差定义为优势函数 $A$: $$ \begin{aligned} &= \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t A^{\pi_\theta}(s_t, a_t) \right] \\ &= \sum_{t=0}^\infty \gamma^t \mathbb{E}_{s_t \sim P^{t}(\cdot|\pi_{\theta'})} \mathbb{E}_{a_t \sim \pi_{\theta'}(\cdot|s_t)} [A^{\pi_\theta}(s_t, a_t)] \\ &= \frac{1}{1-\gamma} \mathbb{E}_{s \sim \nu^{\pi_{\theta'}}} \mathbb{E}_{a \sim \pi_{\theta'}(\cdot|s)} [A^{\pi_\theta}(s, a)] \end{aligned} $$ 最后一个等号的成立运用到了状态访问分布的定义: $\nu^\pi(s) = (1-\gamma) \sum_{t=0}^\infty \gamma^t P^t(s)$, 所以只要我们能找到一个新策略，使得 $\mathbb{E}_{s \sim \nu^{\pi_{\theta'}}} \mathbb{E}_{a \sim \pi_{\theta'}(\cdot|s)} [A^{\pi_\theta}(s, a)] \ge 0$，就能保证策略性能单调递增。 但是直接求解该式是非常困难的，因为我们又需要用它来采集样本。把所有可能的新策略都拿来收集数据，然后判断哪个策略满足上述条件，但我们的做法显然是不现实的。于是TRPO做了一步近似操作，对状态访问分布进行了相应处理，具体而言，忽略两个策略之间的状态访问分布变化，直接采用旧的策略的状态分布，定义如下替代优化目标： $$ L_{\theta}(\theta') = J(\theta) + \frac{1}{1-\gamma} \mathbb{E}_{s \sim \nu^{\pi_\theta}} \mathbb{E}_{a \sim \pi_{\theta'}(\cdot|s)} [A^{\pi_\theta}(s, a)] $$ 当新旧策略非常接近时，状态访问分布变化很小，这么近似是合理的。其中，动作仍然用新策略 $\pi_{\theta'}$ 来采样得到，我们可以用重要性采样对动作分布进行处理： $$ L_{\theta}(\theta') = J(\theta) + \mathbb{E}_{s \sim \nu^{\pi_\theta}} \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \left[ \frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)} A^{\pi_\theta}(s, a) \right] $$ 这样，我们就可以基于旧策略 $\pi_\theta$ 已经采样出的数据来估计并优化新策略 $\pi_{\theta'}$ 了。为了保证新旧策略足够接近，TRPO使用了库尔贝克-莱布勒（Kullback-Leibler, KL）散度来衡量策略之间的距离，并给出了整体的优化公式： $$ \begin{aligned} \max_{\theta'} \quad & L_{\theta}(\theta') \\ \text{s.t.} \quad & \mathbb{E}_{s \sim \nu^{\pi_\theta}} [D_{KL}(\pi_{\theta'}(\cdot|s), \pi_\theta(\cdot|s))] \le \delta \end{aligned} $$ 这里的不等式约束定义了策略空间中的一个KL球，被称为信任区域。在这个区域中，可以认为当前学习策略和环境交互的状态分布与上一个策略最后采样的状态分布一致，进而可以基于一步行动的重要性采样方法使当前策略稳定提升。TRPO背后的原理如下图所示。
## 11.3 近似求解 
直接求解上述带约束的优化问题比较麻烦，TRPO在其具体实现中做了一步近似操作来快速求解。为方便起见，我们在接下来的式子中用 $\theta_k$ 代替之前的 $\theta$，表示这是第 $k$ 次迭代之后的策略。首先对目标函数和约束项在 $\theta_k$ 进行泰勒展开，分别用1阶和2阶近似： $$ \mathbb{E}_{s \sim \nu^{\theta_k}, a \sim \pi_{\theta_k}(\cdot|s)} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a) \right] \approx g^T(\theta' - \theta_k) $$ $$ \mathbb{E}_{s \sim \nu^{\theta_k}} [D_{KL}(\pi_{\theta_k}(\cdot|s), \pi_{\theta'}(\cdot|s))] \approx \frac{1}{2}(\theta' - \theta_k)^T H (\theta' - \theta_k) $$ 其中 $g = \nabla_\theta \mathbb{E}_{s \sim \nu^{\theta_k}, a \sim \pi_{\theta_k}(\cdot|s)} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a) \right] |_{\theta=\theta_k}$ 表示目标函数的梯度， $H = \mathbb{E}_{s \sim \nu^{\theta_k}} [D_{KL}(\pi_{\theta_k}(\cdot|s), \pi_{\theta'}(\cdot|s))]|_{\theta=\theta_k}$ 表示策略之间KL距离的黑塞矩阵（Hessian matrix）。 于是我们的优化目标变成了： $$ \theta_{k+1} = \underset{\theta'}{\text{arg max}} g^T(\theta' - \theta_k) \quad \text{s.t.} \quad \frac{1}{2}(\theta' - \theta_k)^T H (\theta' - \theta_k) \le \delta $$ 此时，我们可以用卡罗需-库恩-塔克（Karush-Kuhn-Tucker, KKT）条件直接导出上述问题的解： $$ \theta_{k+1} = \theta_k + \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g $$
#### 11.4 共轭梯度 
一般说来，用神经网络表示的策略函数的参数数量都是成千上万的，计算和存储黑塞矩阵 $H$ 的逆矩阵会耗费大量的内存资源和时间。TRPO通过共轭梯度法（conjugate gradient method）的回溯线性搜索解决了这个问题，它的核心思想是直接计算 $x=H^{-1}g$，即参数更新方向。假设满足KL距离约束的参数更新时的最大步长为 $\beta$，于是，根据KL距离约束条件，有 $\frac{1}{2}(\beta x)^T H (\beta x) = \delta$。求解 $\beta$，得到了 $\beta = \sqrt{\frac{2\delta}{x^T H x}}$。因此，此时参数更新方式为 $$ \theta_{k+1} = \theta_k + \sqrt{\frac{2\delta}{x^T H x}} x $$ 因此，只要可以直接计算 $x=H^{-1}g$，就可以根据该问题转化为求解 $Hx=g$。实际上只要对称矩阵 $H$ 为正定矩阵，我们就可以使用共轭梯度法来求解。共轭梯度法的具体流程如下： 
* 初始化 $r_0 = g - Hx_0, p_0 = r_0, x_0 = 0$ 
* `for k = 0 -> N do:` 
	* $\alpha_k = \frac{r_k^T r_k}{p_k^T H p_k}$ 
	* $x_{k+1} = x_k + \alpha_k p_k$ 
	* $r_{k+1} = r_k - \alpha_k H p_k$ 
	* 如果 $r_{k+1}^T r_{k+1}$ 非常小，则退出循环 
	* $\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$ 
	* $p_{k+1} = r_{k+1} + \beta_k p_k$ 
* `end for` 
* 输出 $x_{N+1}$ 在共轭梯度运算过程中，直接计算 $\alpha_k$ 和 $r_{k+1}$ 需要计算和存储海森矩阵 $H$。为了避免这种大矩阵的出现，我们只计算 $H$ 乘以向量，而不直接计算和存储 $H$ 矩阵。这样做比较容易，因为对于任意的列向量 $v$，容易验证： $$ Hv = \nabla_\theta \left( (\nabla_\theta D_{KL}^{\pi_{\theta_k}}(\pi_\theta, \pi_{\theta_k}))^T v \right) = \nabla_\theta \left( (\nabla_\theta (\sum_i D_{KL}^{\pi_{\theta_k}}(\pi_\theta, \pi_{\theta_k})_i))^T v \right) $$ 即先用梯度和向量 $v$ 点乘后再计算梯度。 ##11.5 线性搜索 
* 由于TRPO算法用到了泰勒展开的1阶和2阶近似，这并非精准求解，因此，$\theta'$ 可能未必比 $\theta_k$ 好，或未必能满足KL散度限制。TRPO在每次迭代的最后进行一次线性搜索（Line Search），以确保找到满足条件的参数。具体来说，就是找到一个最小的非负整数 $i$，使得按照 $$ \theta_{k+1} = \theta_k + \alpha^i \sqrt{\frac{2\delta}{x^T H x}} x $$ 求出的 $\theta_{k+1}$ 依然满足最初的KL散度限制，并且确实能够提升目标函数 $L_{\theta_k}$，这其中 $\alpha \in (0, 1)$ 是一个决定线性搜索长度的超参数。 
## 11.5 TRPO 算法的大致过程 
* 至此，我们已经基本上清楚了TRPO算法的大致过程，它具体的算法流程如下：
* 初始化策略网络参数 $\theta$，价值网络参数 $\omega$ 
* `for 序列 e = 1 -> E do:` 
	* 用当前策略 $\pi_\theta$ 采样轨迹 $\{s_1, a_1, r_1, s_2, a_2, r_2, \dots\}$ 
	* 根据收集到的数据和价值网络估计每个状态动作对的优势 $A(s_t, a_t)$ 
	* 计算策略目标函数的梯度 $g$ 
	* 用共轭梯度法计算 $x = H^{-1}g$ 
	* 用线性搜索找到一个 $i$ 值，并更新策略网络参数 $\theta_{k+1} = \theta_k + \alpha^i \sqrt{\frac{2\delta}{x^T H x}} x$，其中 $i \in \{1, 2, ..., K\}$ 为能提升策略并满足KL距离限制的最小整数 
	* 更新价值网络参数（与Actor-Critic的更新方法相同） 
* `end for` 
* 
## 11.6 广义优势估计 
* 从11.5节中，我们尚未得知如何估计优势函数 $A$。目前比较常用的一种方法为广义优势估计（Generalized Advantage Estimation, GAE），接下来我们简单介绍一下GAE的做法。首先，用 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 表示时序差分误差，其中 $V$ 是一个已经学习的状态价值函数。于是，根据多步时序差分的思想，有： $$ \begin{aligned} A^{(1)}_t &= \delta_t &&= -V(s_t) + r_t + \gamma V(s_{t+1}) \\ A^{(2)}_t &= \delta_t + \gamma \delta_{t+1} &&= -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) \\ A^{(3)}_t &= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} &&= -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V(s_{t+3}) \\ &\vdots \\ A^{(k)}_t &= \sum_{l=0}^{k-1} \gamma^l \delta_{t+l} &&= -V(s_t) + r_t + \gamma r_{t+1} + \dots + \gamma^{k-1}r_{t+k-1} + \gamma^k V(s_{t+k}) \end{aligned} $$ 然后，GAE将这些不同步数的优势估计进行指数加权平均： $$ \begin{aligned} A_t^{GAE} &= (1-\lambda)(A_t^{(1)} + \lambda A_t^{(2)} + \lambda^2 A_t^{(3)} + \dots) \\ &= (1-\lambda)(\delta_t + \lambda(\delta_t + \gamma \delta_{t+1}) + \lambda^2(\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2}) + \dots) \\ &= (1-\lambda) \left( \delta_t(1+\lambda+\lambda^2+\dots) + \gamma \delta_{t+1}(\lambda+\lambda^2+\lambda^3+\dots) + \gamma^2 \delta_{t+2}(\lambda^2+\lambda^3+\dots) + \dots \right) \\ &= (1-\lambda) \left( \delta_t \frac{1}{1-\lambda} + \gamma \delta_{t+1} \frac{\lambda}{1-\lambda} + \gamma^2 \delta_{t+2} \frac{\lambda^2}{1-\lambda} + \dots \right) \\ &= \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l} \end{aligned} $$ 其中，$\lambda \in [0, 1]$ 是在GAE中额外引入的一个超参数。当 $\lambda=0$ 时，$A_t^{GAE} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，也即仅仅只看一步差分的优势；当 $\lambda=1$ 时，$A_t^{GAE} = \sum_{l=0}^\infty \gamma^l \delta_{t+l} = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$，则是看每一步差分到优势的完全平均值。 下面一段是GAE的代码，给定 $\gamma, \lambda$ 以及每个时间步的 $\delta_t$ 之后，我们可以根据公式直接进行优势估计。 
```python
def compute_advantage(gamma, lmbda, td_delta): 
	 td_delta = td_delta.detach().numpy() 
	 advantage_list = [] 
	 advantage = 0.0 
	 for delta in td_delta\[::-1]: 
		 advantage = gamma * lmbda * advantage + delta 
		 advantage_list.append(advantage) 
	 advantage_list.reverse() 
	 return torch.tensor(advantage_list, dtype=torch.float)
```