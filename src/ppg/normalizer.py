from pathlib import Path
import numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape=(), epsilon=1e-4):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update_from_batch(self, batch):
        """Updates the mean, var and count from a batch of samples."""
        np_batch = np.array(batch)
        batch_mean = np.mean(np_batch, axis=0)
        batch_var = np.var(np_batch, axis=0)
        batch_count = np_batch.shape[0]
        self.update(batch_mean, batch_var, batch_count)

    def update(self, mean, var, count: float):
        delta = mean - self.mean
        tot_count = self.count + count

        m_a = self.var * self.count
        m_b = var * count
        m2 = m_a + m_b + np.square(delta) * self.count * count / tot_count
        
        self.mean = self.mean + delta * count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def norm_obs(self, states, clip: float = 5, epsilon: float = 1e-8):
        # Normalize a set of states.
        norm_ob = (states - self.mean) / np.sqrt(self.var + epsilon),
        return np.clip(norm_ob, -clip, clip)
    
    def save(self, path):
        np.save(
            Path(path) / "normalizer.npy",
            [
                self.mean,
                self.var,
                self.count
            ],
            allow_pickle=True
        )

    def load(self, path):
        norm_obj = np.load(Path(path) / "normalizer.npy", allow_pickle=True)
        self.mean = norm_obj[0]
        self.var = norm_obj[1]
        self.count = norm_obj[2]