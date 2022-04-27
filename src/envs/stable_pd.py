class QTripplet:
    def __init__(self, p: float, v: float, a: float):
        self.pos = p
        self.vel = v
        self.acc = a

    def __str__(self):
        return "(%f, %f, %f)" % (self.pos, self.vel, self.acc)


class StablePD:
    def __init__(self, kp: float = 0.9, kd: float = 0.0001, delta_time: float = None):
        """
        Create a stable proportional derivative controller
        """
        self.kp = kp
        self.kd = kd
        self.delta_time = delta_time

    def control_with_vel(
        self,
        state: QTripplet,
        next_target_pos: float,
        next_target_vel: float,
        delta_time: float = None,
    ):
        if delta_time is None:
            delta_time = self.delta_time

        control_p = -self.kp * (state.pos + delta_time * state.vel - next_target_pos)
        control_d = -self.kd * (state.vel + delta_time * state.acc - next_target_vel)
        return control_p + control_d

    def control(self, state: QTripplet, next_target: float, delta_time: float = None):
        if delta_time is None:
            delta_time = self.delta_time

        control_p = -self.kp * (state.pos + delta_time * state.vel - next_target)
        control_d = -self.kd * (state.vel + delta_time * state.acc)
        return control_p + control_d
