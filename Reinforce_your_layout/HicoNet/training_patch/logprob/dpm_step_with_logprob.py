from typing import Optional, Tuple, Union
import math
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def dpm_step_with_logprob(
    self: DPMSolverMultistepScheduler,
    model_output: torch.Tensor,
    timestep: Union[int, torch.Tensor],
    sample: torch.Tensor,
    s_churn: float = 0.05,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
    inference_dtype = torch.float32,
) -> Union[SchedulerOutput, Tuple]:

    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    # Improve numerical stability for small number of steps
    lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
        self.config.euler_at_final
        or (self.config.lower_order_final and len(self.timesteps) < 15)
        or self.config.final_sigmas_type == "zero"
    )
    lower_order_second = (
        (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
    )

    model_output = self.convert_model_output(model_output, sample=sample)
    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
    self.model_outputs[-1] = model_output

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(inference_dtype)

    std_dev_t = torch.tensor(s_churn, device=model_output.device, dtype=model_output.dtype)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if prev_sample is None:
        noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=inference_dtype
        )
    else:
        noise = prev_sample.to(device=model_output.device, dtype=inference_dtype)

    # if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
    #     prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
    # elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
    #     prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
    # else:
    #     prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample, noise=noise)

    prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)

    if self.lower_order_nums < self.config.solver_order:
        self.lower_order_nums += 1

    # Cast sample back to expected dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    self._step_index += 1

    log_prob = (
            -((sample - prev_sample) ** 2) / (2 * (std_dev_t ** 2))
            - torch.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob