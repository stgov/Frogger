import gymnasium as gym


class FireResetEnv(gym.Wrapper):
    """
    Presiona automáticamente la acción FIRE al reiniciar el entorno
    y cada vez que el agente pierde una vida en Breakout.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Verificar que el entorno tenga la acción FIRE en el índice 1
        action_meanings = env.unwrapped.get_action_meanings()
        assert len(action_meanings) > 1 and action_meanings[1] == "FIRE", (
            "Este wrapper requiere que la acción 1 sea 'FIRE'."
        )
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        self.was_real_done = True
        obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()

        # Ejecuta FIRE para sacar la pelota al inicio
        obs, _, terminated, truncated, _ = self.env.step(1)  # 1 = FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            return obs, info

        # Ejecuta un paso neutro (o movimiento) para asegurar que el motor procese el saque
        obs, _, terminated, truncated, info = self.env.step(2)  # 2 = RIGHT
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = self.env.unwrapped.ale.lives()

        # Si perdió una vida pero la partida no terminó, presiona FIRE automáticamente
        if 0 < current_lives < self.lives and not (terminated or truncated):
            self.lives = current_lives
            obs, fire_reward, terminated, truncated, info = self.env.step(1)  # Presiona FIRE
            reward += fire_reward

        self.lives = current_lives
        return obs, reward, terminated, truncated, info
