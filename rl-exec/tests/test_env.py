from rl_exec.env import ExecEnv, twap, frontload

def test_twap_vs_frontload():
    env=ExecEnv(seed=42)
    c_twap=twap(env)
    env=ExecEnv(seed=42)
    c_front=frontload(env)
    # Under this parametrization, TWAP should generally be cheaper than naive frontload
    assert c_twap <= c_front
