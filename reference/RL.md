
### 模仿学习
获取先验

### PG(Policy Gradient)

在 PG 算法中，我们的 Agent 又被称为 Actor

$$\tau = \{s_1,a_1,s_2,a_2,\cdots,s_T,a_T\}$$

因为在给定状态下智能体根据策略采取那个动作是不确定的，而且在某一个状态下采取了某一个动作后，由该状态会跳转哪一个状态也是不确定的。

$$p_{\theta}(\tau) = p(s_1)p_{\theta}(a_1|s_1)p(s_2|s_1,a_1)p_{\theta}(a_2|s_2)p(s_3|s_3,a_2)\cdots$$

$$p_{\theta}(\tau) = p(s_1) \prod_{t=1}^T p_{\theta}(a_t|s$$

序列$\tau$ 所获得的奖励为每个阶段所得到的的奖励之和，称之为$R(\tau)$。因此 Actor 的策略为 $\pi$ 的情况下，所能够获得的期望奖励为:

$$\nabla \vec{R}_{\theta} = \sum_{\tau} R(\tau) \nabla p_{\theta}(\tau) = \sum_{\tau} R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)}  $$


$$\nabla f(x) = f(x)\nabla f(x)$$

$$\nabla \vec{R}_{\theta} = \sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla p_{\theta}(\tau)$$

$$\nabla \vec{R}_{\theta} = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau) \nabla \log p_{\theta}(\tau)] \approx \frac{1}{N} \sum_{n=1}^N R(\tau^n) \nabla \log p_{\theta}(\tau^n)$$

$$\vec{R}_{\theta} = \sum_{\tau} R(\tau)p_{\theta}(\tau) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau)] $$

### PPO(Proximal Policy Optimization) 算法原理及实现

