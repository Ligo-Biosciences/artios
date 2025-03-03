# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch


class EMAWrapper(object):
    """A wrapper class for exponential moving average of model weights."""

    def __init__(
        self, model: torch.nn.Module, decay: float = 0.999, mutable_param_keywords=None
    ):
        """
        model: a pytorch model to apply EMA
        decay: a scaler to indicate the decay rate
        mutable_param_keywords: keywords of parameters to apply EMA decay, other params will stay untouched
        """
        self.model = model
        self.decay = decay
        self.mutable_param_keywords = [
            s.strip() for s in mutable_param_keywords if s.strip()
        ]
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if self.mutable_param_keywords and not any(
                [keyword in name for keyword in self.mutable_param_keywords]
            ):
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[
                name
            ]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            assert name in self.shadow
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}


class PowerLawEMAWrapper(EMAWrapper):
    """"
    A wrapper class for EMA that uses a power law decay. See equation (119) and (127) of 
    Karras et al. "Analyzing and Improving the Training Dynamics of Diffusion Models"
    for more details.
    """
    def __init__(
            self, 
            model: torch.nn.Module, 
            gamma: float = 6.94, 
            delta_t: int = 1000,
            mutable_param_keywords=None):
        """
        model: (torch.nn.Module) a pytorch model to apply power law EMA
        gamma: (float) decay rate for the power law
        delta_t: (int) the number of steps between updates. This is the evaluation interval.
        mutable_param_keywords: (list[str]) keywords of parameters to apply EMA decay, other params will stay untouched
        """
        self.model = model
        self.gamma = gamma
        self.delta_t = delta_t

        self.mutable_param_keywords = [
            s.strip() for s in mutable_param_keywords if s.strip()
        ]
        self.shadow = {}
        self.backup = {}

    def beta_power_law(self, t):
        """Calculate the timestep-dependent power law beta parameter."""
        #return 0 if t_next == self.delta_t else (1 - self.delta_t / t_next) ** (self.gamma + 1)
        return 0 if t == 1 else (1 - 1 / t) ** (self.gamma + 1)
    
    def update(self, t):
        """
        Update the EMA weights using the power law decay.
        
        Args:
            t (int): The current training step.
        """
        for name, param in self.model.named_parameters():
            if self.mutable_param_keywords and not any(
                [keyword in name for keyword in self.mutable_param_keywords]
            ):
                continue
            assert name in self.shadow

            beta = self.beta_power_law(t)
            if isinstance(beta, complex) or (isinstance(beta, float) and not np.isreal(beta)):
                raise ValueError(f"Beta value {beta} is complex at step {t}. This should not happen with valid gamma={self.gamma} and delta_t={self.delta_t}")
            new_average = (1.0 - beta) * param.data + beta * self.shadow[name]
            self.shadow[name] = new_average.to(param.dtype)  # Ensure same dtype as parameter

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            assert name in self.shadow
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            assert name in self.backup
            param.data = self.backup[name]
        self.backup = {}


def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    """
    Calculates the matrix of dot product between two power law decay functions with 
    respective decay rates gamma_a and gamma_b and respective checkpoint times t_a and t_b.

    Args:
        t_a (np.array): Array of checkpoint times for the first decay function.
        gamma_a (np.array): The decay rate for the first decay function.
        t_b (np.array): Array of checkpoint times for the second decay function.
        gamma_b (np.array): The decay rate for the second decay function.

    Returns:
        np.array: Matrix of dot products between the two decay functions. 
            [len(t_a), len(t_b)]
    """
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = np.maximum(t_a, t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp 
    den = (gamma_a + gamma_b + 1) * t_max 
    return num / den


def solve_weights(t_i, gamma_i, gamma_r):
    """
    Solves for the optimal weights x_i in the post hoc EMA calculation. 
    So, for each gamma value in gamma_r, and for input checkpoints taken at times t_i with decay rates gamma_i, we
    solve for the optimal weights x_i that minimize the error between the input checkpoints and the post hoc EMA profile.
    For simplicity, we only calculate the post hoc EMA profile for the final checkpoint time t_c = t_i[-1] 
    (see equation 107 of Karras et al.).
     
    Args:
        t_i (list[int]): Array of checkpoint times for the training time checkpoints.
        gamma_i (list[float]): Array of decay rates used for training time checkpoints.
        gamma_r (list[float]): Array of decay rates for the post hoc EMA profiles.

    Returns:
        np.array: Array of weights for the first decay function. The columns correspond to the gamma_r values 
        and the rows correspond to the training checkpoints.
            [len(t_i)*len(gamma_i), len(gamma_r)]
    """
    # Convert inputs to numpy arrays
    t_i = np.array([t_i]).reshape(1, -1)  # [1, len(t_i)]
    gamma_i = np.array(gamma_i).reshape(-1, 1)  # [len(gamma_i), 1]
    gamma_r = np.array(gamma_r).reshape(-1, 1)  # [len(gamma_r), 1]
    t_r = np.array([t_i[0, -1]]).reshape(1, -1)  # Use final time from t_i

    # Broadcast arrays to match shapes
    t_i_broadcast, gamma_i_broadcast = np.broadcast_arrays(t_i, gamma_i)
    t_r_broadcast, gamma_r_broadcast = np.broadcast_arrays(t_r, gamma_r)

    # Convert to column/row vectors
    rv = lambda x: np.float64(x).reshape(-1, 1)  
    cv = lambda x: np.float64(x).reshape(1, -1)  

    t_i_row = rv(t_i_broadcast)
    gamma_i_row = rv(gamma_i_broadcast) 
    t_i_col = cv(t_i_broadcast)
    gamma_i_col = cv(gamma_i_broadcast)
    t_r_col = cv(t_r_broadcast)
    gamma_r_col = cv(gamma_r_broadcast)

    # Create matrices A and B (see equation 142 and 143 of Karras et al.)
    A = p_dot_p(t_i_row, gamma_i_row, t_i_col, gamma_i_col)
    B = p_dot_p(t_i_row, gamma_i_row, t_r_col, gamma_r_col)

    # Solve system for each gamma_r
    return np.linalg.solve(A, B)

