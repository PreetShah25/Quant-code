from __future__ import annotations
import numpy as np

class ExecEnv:
    """Simple discrete-time execution environment.
    State: (t, q_remaining, price)
    Action: shares to sell this step (0..max_per_step)
    Price dynamics: dP = sigma*eps - gamma*action (perm impact)
    Temporary impact: cost += eta*action^2
    """
    def __init__(self, horizon=20, q0=10_000, max_per_step=1_000, p0=100.0, sigma=0.2, gamma=1e-4, eta=5e-5, seed=0):
        self.horizon=horizon; self.q0=q0; self.max_per_step=max_per_step
        self.p0=p0; self.sigma=sigma; self.gamma=gamma; self.eta=eta
        self.rng=np.random.default_rng(seed)
        self.reset()
    def reset(self):
        self.t=0; self.q=self.q0; self.p=self.p0; self.cost=0.0
        return (self.t, self.q, self.p)
    def step(self, action:int):
        action=int(np.clip(action,0,min(self.max_per_step,self.q)))
        # execution price with temp impact
        exec_price=self.p - self.eta*action
        self.cost += action*exec_price
        self.q -= action
        # price evolution with perm impact and noise
        eps=self.rng.normal(0,self.sigma)
        self.p = self.p + eps - self.gamma*action
        self.t += 1
        done = (self.t>=self.horizon) or (self.q==0)
        return (self.t, self.q, self.p), done

def twap(env: ExecEnv):
    env.reset()
    per_step = int(np.ceil(env.q0/env.horizon))
    while True:
        _,done = env.step(per_step)
        if done:
            # if leftover shares, sell them at last step
            if env.q>0:
                env.step(env.q)
            break
    return env.cost

def frontload(env: ExecEnv):
    env.reset()
    # sell as much as possible early
    while env.q>0:
        _,done=env.step(env.max_per_step)
        if done: break
    return env.cost
