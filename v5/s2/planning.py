





# from https://github.com/vladisai/PLDM/blob/main/pldm/planning/mpc.py







from pldm.planning import objectives_v2
import torch
from pldm.models.jepa import JEPA
from pldm_envs.utils.normalizer import Normalizer
from pldm.planning.planners.enums import PlannerType
from pldm.planning.planners.mppi_planner import MPPIPlanner
from pldm.planning.planners.sgd_planner import SGDPlanner
from pldm.planning.utils import normalize_actions
from abc import ABC
from pldm.planning.enums import MPCResult, PooledMPCResult
import numpy as np
from pldm.models.utils import flatten_conv_output
from tqdm import tqdm


class MPCEvaluator(ABC):
    def __init__(
        self,
        config,
        model: JEPA,
        prober: torch.nn.Module,
        normalizer: Normalizer,
        quick_debug: bool = False,
        prefix: str = "",
        pixel_mapper=None,
        image_based=True,
    ):
        self.config = config
        self.model = model
        self.prober = prober
        self.normalizer = normalizer
        self.quick_debug = quick_debug
        self.prefix = prefix
        self.pixel_mapper = pixel_mapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_based = image_based

    def close(self):
        pass

    def _infer_chunk_sizes(self):
        config = self.config

        if config.level1.planner_type == PlannerType.MPPI:
            # n_envs_batch_size = 500000 // (config.num_samples * config.max_plan_length)
            n_envs_batch_size = config.n_envs_batch_size
        else:
            n_envs_batch_size = config.n_envs

        chunk_sizes = [n_envs_batch_size] * (config.n_envs // n_envs_batch_size) + (
            [config.n_envs % n_envs_batch_size]
            if config.n_envs % n_envs_batch_size != 0
            else []
        )

        return chunk_sizes

    def _construct_planner(self, n_envs: int):
        config = self.config

        objective = objectives_v2.ReprTargetMPCObjective(
            model=self.model,
            propio_cost=config.level1.propio_cost,
            sum_all_diffs=config.level1.sum_all_diffs,
            loss_coeff_first=config.level1.loss_coeff_first,
            loss_coeff_last=config.level1.loss_coeff_last,
        )

        action_normalizer = lambda x: normalize_actions(
            x,
            min_norm=config.level1.min_step,
            max_norm=config.level1.max_step,
            xy_action=True,
            clamp_actions=config.level1.clamp_actions,
        )

        if config.level1.planner_type == PlannerType.MPPI:
            planner = MPPIPlanner(
                config.level1.mppi,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,  # [300, 13456]
                prober=self.prober,
                action_normalizer=action_normalizer,
                n_envs=n_envs,
                projected_cost=config.level1.projected_cost,
            )
        elif config.level1.planner_type == PlannerType.SGD:
            planner = SGDPlanner(
                config.level1.sgd,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,
                prober=self.prober,
                action_normalizer=action_normalizer,
            )
        else:
            raise NotImplementedError(
                f"Unknown planner type {config.level1.planner_type}"
            )

        return planner

    def _perform_mpc_in_chunks(self):
        """
        Divide it up in chunks in order to prevent OOM
        """
        chunk_sizes = self._infer_chunk_sizes()

        mpc_data = PooledMPCResult()
        chunk_offset = 0

        for chunk_size in chunk_sizes:
            planner = self._construct_planner(n_envs=chunk_size)
            envs = self.envs[chunk_offset : chunk_offset + chunk_size]

            mpc_result = self._perform_mpc(
                planner=planner,
                envs=envs,
            )

            obs_c = mpc_result.observations
            location_history_c = mpc_result.locations
            action_history_c = mpc_result.action_history
            reward_history_c = mpc_result.reward_history
            pred_locations_c = mpc_result.pred_locations
            final_preds_dist_c = mpc_result.final_preds_dist
            targets_c = mpc_result.targets
            loss_history_c = mpc_result.loss_history
            qpos_history_c = mpc_result.qpos_history
            propio_history_c = mpc_result.propio_history

            mpc_data.observations.append(obs_c)
            mpc_data.locations.append(location_history_c)
            mpc_data.action_history.append(action_history_c)
            mpc_data.reward_history.append(reward_history_c)
            mpc_data.pred_locations.append(pred_locations_c)
            mpc_data.final_preds_dist.append(final_preds_dist_c)
            mpc_data.targets.append(targets_c)
            mpc_data.loss_history.append(loss_history_c)
            mpc_data.qpos_history.append(qpos_history_c)
            mpc_data.propio_history.append(propio_history_c)

            chunk_offset += chunk_size

        mpc_data.concatenate_chunks()

        return mpc_data

    def _perform_mpc(
        self,
        planner,
        envs,
    ):
        """
        Parameters:
            starts: (bs, 4)
            targets: (bs, 4)
        Outputs:
            observations: list of a_T (bs, 3, 64, 64) or (bs, 2)
            locations: list of a_T (bs, 2)
            action_history: list of a_T (bs, p_T, 2)
            reward_history: list of a_T (bs,)
            pred_locations: list of a_T (p_T, bs, 1, 2)
            targets: (bs, 4)
            loss_history: list of a_T (n_iters,)
        """

        targets = [e.get_target() for e in envs]
        targets = torch.from_numpy(np.stack(targets))

        targets_t = torch.stack([e.get_target_obs() for e in envs]).to(self.device)

        # encode target obs
        if self.model.config.backbone.propio_dim is not None:
            # for target we don't care about the proprioceptive states. just make it zero.
            propio_states = torch.zeros(
                (targets_t.shape[0], self.model.config.backbone.propio_dim)
            ).to(self.device)
            targets_t = self.model.backbone(
                targets_t, propio=propio_states
            ).obs_component.detach()
        else:
            targets_t = self.model.backbone(targets_t).obs_component.detach()

        targets_t = flatten_conv_output(targets_t)
        planner.reset_targets(targets_t, repr_input=True)

        observation_history = [torch.stack([e.get_obs() for e in envs])]

        obs_t = observation_history[0]
        if self.image_based:
            obs_t = torch.cat([obs_t] * self.config.stack_states, dim=1)  # VERIFY

        action_history = []
        reward_history = []
        location_history = []
        qpos_history = []
        propio_history = []

        pred_positions_history = []
        loss_history = []
        final_preds_dist_history = []

        init_infos = [e.get_info() for e in envs]
        if "location" in init_infos[0]:
            location_history.append(np.array([info["location"] for info in init_infos]))

        if "qpos" in init_infos[0]:
            qpos_history.append(np.array([info["qpos"] for info in init_infos]))

        if "propio" in init_infos[0]:
            propio_history.append(np.array([info["propio"] for info in init_infos]))

        for i in tqdm(range(self.config.n_steps), desc="Planning steps"):
            if i % self.config.replan_every == 0:

                if planner.model.use_propio_pos:
                    curr_propio_pos = [e.get_propio_pos(normalized=True) for e in envs]
                    curr_propio_pos = torch.from_numpy(
                        np.stack(curr_propio_pos)
                    ).float()
                else:
                    curr_propio_pos = None

                if planner.model.use_propio_vel:
                    curr_propio_vel = [e.get_propio_vel(normalized=True) for e in envs]
                    curr_propio_vel = torch.from_numpy(
                        np.stack(curr_propio_vel)
                    ).float()
                else:
                    curr_propio_vel = None

                planning_result = planner.plan(
                    obs_t,
                    curr_propio_pos=curr_propio_pos,
                    curr_propio_vel=curr_propio_vel,
                    plan_size=min(
                        self.config.n_steps - i, self.config.level1.max_plan_length
                    ),
                    repr_input=False,
                )

            last_pred_obs = flatten_conv_output(planning_result.pred_obs)

            pred_dist = torch.norm(last_pred_obs - targets_t.unsqueeze(0), dim=2).cpu()
            final_preds_dist_history.append(pred_dist)

            planned_actions = (
                planning_result.actions[:, i % self.config.replan_every :]
                .detach()
                .cpu()
            )

            if self.config.random_actions:
                results = [
                    envs[j].step(envs[0].action_space.sample())
                    for j in range(len(envs))
                ]
            else:
                results = [
                    envs[j].step(
                        planned_actions[j, 0].detach().cpu().contiguous().numpy()
                    )
                    for j in range(len(envs))
                ]

            assert len(results[0]) == 5
            current_obs = torch.from_numpy(np.stack([r[0] for r in results])).float()
            rewards_t = torch.from_numpy(np.stack([r[1] for r in results])).float()
            infos = [r[4] for r in results]

            action_history.append(planned_actions.detach().cpu())
            observation_history.append(current_obs)
            reward_history.append(rewards_t)

            if "location" in infos[0]:
                location_history.append(np.array([info["location"] for info in infos]))

            if "qpos" in infos[0]:
                qpos_history.append(np.array([info["qpos"] for info in infos]))

            if "propio" in infos[0]:
                propio_history.append(np.array([info["propio"] for info in infos]))

            if planning_result.locations is not None:
                pred_locations = planning_result.locations.detach().cpu()
                pred_locations = pred_locations.squeeze(2)
                pred_positions_history.append(pred_locations)

            # stack states if necessary for next iteration
            if self.config.stack_states == 1:
                obs_t = current_obs
            else:
                obs_t = torch.cat(
                    [obs_t[:, current_obs.shape[1] :], current_obs], dim=1
                )

            loss_history.append(planning_result.losses)

        observation_history = [
            self.normalizer.unnormalize_state(o) for o in observation_history
        ]

        return MPCResult(
            observations=observation_history,
            locations=[torch.from_numpy(x) for x in location_history],
            action_history=action_history,
            reward_history=reward_history,
            pred_locations=pred_positions_history,
            final_preds_dist=final_preds_dist_history,
            targets=targets,
            loss_history=loss_history,
            qpos_history=[torch.from_numpy(x) for x in qpos_history],
            propio_history=[torch.from_numpy(x) for x in propio_history],
        )




















# -------------------------------------------------------------------




















# from https://github.com/facebookresearch/jepa-wms/blob/main/evals/simu_env_planning/planning/planning/planner.py










# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple

import nevergrad as ng
import numpy as np
import torch
import torch.distributed as dist

from evals.simu_env_planning.planning.planning import objectives
from src.utils.logging import get_logger

logger = get_logger(__name__)

########### PLANNERS IN LATENT SPACE ###############


class PlanningResult(NamedTuple):
    actions: torch.Tensor
    # locations that the model has planned to achieve
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None
    plan_metrics: dict = None
    pred_frames_over_iterations: List = None
    predicted_best_encs_over_iterations: List = None


class Planner(ABC):
    def __init__(self, unroll: Callable):
        self.objective = None
        self.unroll = unroll

    def set_objective(self, objective: objectives.BaseMPCObjective):
        self.objective = objective

    @abstractmethod
    def plan(self, obs: torch.Tensor, steps_left: int):
        pass

    def cost_function(self, actions: torch.Tensor, z_init: torch.Tensor) -> torch.Tensor:
        predicted_encs = self.unroll(z_init, actions)
        return self.objective(predicted_encs, actions)


class NevergradPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        iterations: int,
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_samples: int = 1,
        horizon: int = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        num_elites: int = 10,
        optimizer_name: str = "NgIohTuned",
        **kwargs,
    ):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.iterations = iterations
        self.var_scale = var_scale
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.num_samples = num_samples
        self.horizon = horizon
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.num_elites = num_elites  # just for logging
        self.optimizer_name = optimizer_name
        self.optimizer_map = {
            "NgIohTuned": ng.optimizers.NgIohTuned,
            "NGOpt": ng.optimizers.NGOpt,
            # CMA-ES variants - numerically stable, good for continuous optimization
            "CMA": ng.optimizers.CMA,
            "ParametrizedCMA": ng.optimizers.ParametrizedCMA,
            "DiagonalCMA": ng.optimizers.DiagonalCMA,
            # Other stable alternatives
            "PSO": ng.optimizers.PSO,
            "DE": ng.optimizers.DE,
            "OnePlusOne": ng.optimizers.OnePlusOne,
            "TwoPointsDE": ng.optimizers.TwoPointsDE,
        }

    def build_optimizer(self, optimizer_name, **kwargs):
        """Build an optimizer by name."""
        if optimizer_name in self.optimizer_map:
            return self.optimizer_map[optimizer_name](**kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_optimizer(self, plan_length: int):
        parametrization = ng.p.Array(shape=(self.horizon, self.action_dim))
        if self.max_norms is not None:
            lower_bounds = -np.ones((plan_length, self.action_dim))
            upper_bounds = np.ones((plan_length, self.action_dim))

            for max_norm_group, dims in zip(self.max_norms, self.max_norm_dims):
                for d in dims:
                    lower_bounds[:, d] = -max_norm_group
                    upper_bounds[:, d] = max_norm_group

            parametrization.set_bounds(lower=lower_bounds, upper=upper_bounds)
        optimizer = self.build_optimizer(
            self.optimizer_name,
            parametrization=parametrization,
            budget=self.iterations * self.num_samples,
            num_workers=self.num_samples,
        )
        logger.info(f"⚙️  Optimizer: {optimizer}")
        logger.info(f"   Optimizer info: {optimizer._info()}")

        # Check if NGOpt selected MetaModel - it causes numerical instability
        # due to polynomial regression overflow when loss variance is low.
        # In this case, replace with DiagonalCMA which is what NGOpt typically
        # selects in other configurations and is more numerically stable.
        if hasattr(optimizer, "optim") and optimizer.optim.name == "MetaModel":
            logger.warning(
                "NGOpt selected MetaModel optimizer which can cause numerical instability. "
                "Switching to DiagonalCMA for better numerical stability."
            )
            optimizer = self.build_optimizer(
                "DiagonalCMA",
                parametrization=parametrization,
                budget=self.iterations * self.num_samples,
                num_workers=self.num_samples,
            )
            logger.info(f"⚙️  Replacement optimizer: {optimizer}")

        if hasattr(optimizer, "optim"):
            if optimizer.optim.name in ["MetaModel", "CMApara"]:
                if hasattr(optimizer.optim, "_optim"):
                    if hasattr(optimizer.optim._optim, "_es") and optimizer.optim._optim._es is not None:
                        logger.info(f"{optimizer.optim._optim._es.inopts=}")
                    else:
                        logger.info("No _es in optimizer")
        return optimizer

    @torch.no_grad()
    def plan(
        self,
        z_init: torch.Tensor,
        steps_left: int = None,
    ) -> PlanningResult:
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon
        optimizer = self._get_optimizer(plan_length)
        costs = []
        prev_elite_losses_mean = []
        prev_elite_losses_std = []
        pred_frames_over_iterations = []
        predicted_best_encs_over_iterations = []

        for itr in range(self.iterations):
            candidates = [optimizer.ask() for _ in range(self.num_samples)]
            candidate_values = torch.from_numpy(np.array([c.value for c in candidates])).to(
                device=z_init.device, dtype=torch.float32
            )
            loss = self.cost_function(candidate_values.permute(1, 0, 2), z_init)

            # Check for NaN or Inf values in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"NaN or Inf detected in loss at iteration {itr}. Replacing with large values.")
                loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=-1e6)

            # for logging
            elite_losses = torch.topk(loss, k=self.num_elites, largest=False).values
            prev_elite_losses_mean.append(elite_losses.mean().item())
            prev_elite_losses_std.append(elite_losses.std().item())

            for i, c in enumerate(candidates):
                optimizer.tell(c, loss[i].item())
            costs.append(loss.min().item())

            best_solution = optimizer.provide_recommendation().value
            actions = torch.tensor(best_solution, device=z_init.device, dtype=torch.float32).unsqueeze(1)
            predicted_best_encs = self.unroll(z_init, act_suffix=actions)
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(predicted_best_encs)
                pred_frames_over_iterations.append(pred_frames)

        best_solution = optimizer.provide_recommendation().value
        actions = torch.tensor(best_solution, device=z_init.device)
        result = PlanningResult(
            actions=actions[: self.num_act_stepped],
            losses=torch.tensor(costs).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(prev_elite_losses_mean).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(prev_elite_losses_std).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class CEMPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        iterations: int = 6,
        num_samples: int = 512,
        horizon: int = 32,
        action_dim: int = 4,
        var_scale: float = 1,
        num_elites: int = 64,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        distribute_planner: bool = False,
        local_generator: torch.Generator = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.iterations = iterations
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = torch.device("cuda")
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.momentum_mean = momentum_mean
        self.momentum_std = momentum_std
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self._prev_mean = None
        self.distribute_planner = distribute_planner
        self.local_generator = local_generator
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll

    @torch.no_grad()
    def plan(
        self,
        z_init,
        steps_left=None,
    ):
        """
        Same as MPPIPlanner but without a policy network.
        Plan a sequence of actions using the learned world model.
        This planner assumes independence between temporal dimensions: we sample actions according
        to a diagonal Gaussian

        Args:
                z_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.horizon
        else:
            plan_length = min(self.horizon, steps_left)
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )
        losses, elite_means, elite_stds = [], [], []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # Iterate CEM
        for itr in range(self.iterations):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=std.device, generator=self.local_generator
            )
            # Mean sample inclusion trick to never loose best previous action
            actions[:, 0, :] = mean
            # Apply clipping if max_norms is specified
            if self.max_norms is not None:
                for h in range(plan_length):
                    # Loop through each group of dimensions to clip
                    for i, (dims, maxnorm) in enumerate(zip(self.max_norm_dims, self.max_norms)):
                        # Clip the specified dimensions to [-maxnorm, maxnorm]
                        actions[h, :, dims] = torch.clip(actions[h, :, dims], min=-maxnorm, max=maxnorm)
            # Compute elite actions
            cost = self.cost_function(actions, z_init).unsqueeze(1)
            losses.append(cost.min().item())
            # Gather all values
            if self.distribute_planner:
                cost = torch.cat(FullGatherLayer.apply(cost), dim=0)
                all_actions = torch.cat(FullGatherLayer.apply(actions), dim=1)
            else:
                all_actions = actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], all_actions[:, elite_idxs]  # [EL,1] , [H,EL,A]
            # Log the mean and std of the elite values
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())
            # Update parameters with momentum
            new_mean = torch.mean(elite_actions, dim=1)
            new_std = torch.std(elite_actions, dim=1)
            # Apply momentum to mean and std updates
            mean = new_mean * (1 - self.momentum_mean) + mean * self.momentum_mean
            std = new_std * (1 - self.momentum_std) + std * self.momentum_std
            # Decoding logic
            predicted_best_encs = self.unroll(z_init, act_suffix=mean.unsqueeze(1))
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames)
                # [T H W 3]: uint 8 in [0, 255]

        self._prev_mean = mean
        a = mean[: self.num_act_stepped]
        if self.distribute_planner:
            dist.broadcast(a, src=0)
        result = PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class MPPIPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        iterations: int = 6,
        num_samples: int = 512,
        horizon: int = 32,
        action_dim: int = 4,
        max_std: float = 2,
        min_std: float = 0.05,
        num_elites: int = 64,
        temperature: float = 0.5,
        distribute_planner: bool = False,
        local_generator: torch.Generator = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.iterations = iterations
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = torch.device("cuda")
        self.max_std = max_std
        self.min_std = min_std
        self.num_elites = num_elites
        self.temperature = temperature
        self._prev_mean = None
        self.distribute_planner = distribute_planner
        self.local_generator = local_generator
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll

    @torch.no_grad()
    def plan(self, z_init, eval_mode=False, task=None, steps_left=None):
        """
        MPPIPlanner without a policy network.
        Plan a sequence of actions using the learned world model.

        Args:
                z_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.horizon
        else:
            plan_length = min(self.horizon, steps_left)

        # Initialize state and parameters
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses, elite_means, elite_stds = [], [], []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # Iterate MPPI
        for _ in range(self.iterations):
            # Sample actions
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
                generator=self.local_generator,
            )
            # Compute costs
            cost = self.cost_function(actions, z_init).unsqueeze(1)
            losses.append(cost.min().item())
            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())
            # Update parameters
            min_cost = cost.min(0)[0]
            score = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))  # increasing with elite_value
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1) / (score.sum(0) + 1e-9)  # T B A
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2) * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,  # T B A
                )
                / (score.sum(0) + 1e-9)
            )
            # Decoding logic
            predicted_best_encs = self.unroll(z_init, act_suffix=mean.unsqueeze(1))
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames)
                # [T H W 3]: uint 8 in [0, 255]
        # Select action
        score = score.cpu().numpy()  # [EL,]
        # actions: [H, A]
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]  # [H,A]
        self._prev_mean = mean
        a, std = actions[: self.num_act_stepped], std[: self.num_act_stepped]  # [N, A], [N, A]
        if not eval_mode:
            a += std * torch.randn(self.action_dim, device=std.device, generator=self.local_generator)
        # to make sure each GPU outputs same action
        if self.distribute_planner:
            dist.broadcast(a, src=0)

        result = PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class GradientDescentPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int,
        iterations: int = 500,
        lr: float = 1,
        action_noise: float = 0.003,
        sample_type: str = "randn",
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        optimizer_type: str = "sgd",
        adam_betas: tuple = (0.9, 0.995),
        adam_eps: float = 1e-8,
        **kwargs,
    ):
        """
        Gradient Descent Planner for action optimization in latent space.

        Args:
            unroll: Function to unroll the world model
            action_dim: Dimension of the action space
            horizon: Planning horizon (number of timesteps)
            iterations: Number of optimization iterations
            lr: Learning rate for gradient descent
            action_noise: Standard deviation of Gaussian noise to add after each gradient step
            sample_type: Type of action initialization ("randn" or "zero")
            max_norms: List of maximum norm values for each group of dimensions (None to disable clipping)
            max_norm_dims: List of dimension groups to clip (e.g., [[0, 1, 2], [6]])
            num_act_stepped: Number of actions to execute (default: all)
            decode_each_iteration: Whether to decode predictions at each iteration
            decode_unroll: Function to decode latent predictions to frames
            optimizer_type: Type of optimizer to use ("sgd" or "adam")
            adam_betas: Betas for Adam optimizer (default: (0.9, 0.995))
            adam_eps: Epsilon for Adam optimizer (default: 1e-8)
        """
        super().__init__(unroll)
        self.action_dim = action_dim
        self.horizon = horizon
        self.iterations = iterations
        self.lr = lr
        self.action_noise = action_noise
        self.var_scale = var_scale
        self.sample_type = sample_type
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.optimizer_type = optimizer_type.lower()
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.device = torch.device("cuda")

    def init_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize actions for planning.

        Args:
            device: Device to place actions on

        Returns:
            actions: (1, horizon, action_dim) initialized actions
        """
        if self.sample_type == "randn":
            actions = torch.randn(1, self.horizon, self.action_dim, device=device) * self.var_scale
        elif self.sample_type == "zero":
            actions = torch.zeros(1, self.horizon, self.action_dim, device=device)
        else:
            raise ValueError(f"Unknown sample_type: {self.sample_type}")
        return actions

    def plan(
        self,
        z_init: torch.Tensor,
        steps_left: int = None,
    ) -> PlanningResult:
        """
        Plan a sequence of actions using gradient descent optimization.

        Args:
            z_init: Initial latent state
            steps_left: Number of steps left in episode (optional)

        Returns:
            PlanningResult with optimized actions and planning metrics
        """
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon

        # Initialize actions: (batch_size, plan_length, action_dim)
        actions = self.init_actions(1, self.device)[:, :plan_length, :]
        actions.requires_grad = True

        # Setup optimizer based on optimizer_type
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam([actions], lr=self.lr, betas=self.adam_betas, eps=self.adam_eps)
        else:
            optimizer = torch.optim.SGD([actions], lr=self.lr)

        losses = []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []

        # Optimization loop
        for itr in range(self.iterations):
            optimizer.zero_grad()

            # Unroll world model with current actions
            # actions shape: (1, plan_length, action_dim)
            # Need to transpose to (plan_length, 1, action_dim) for unroll
            actions_transposed = actions.transpose(0, 1)

            predicted_encs = self.unroll(z_init, act_suffix=actions_transposed)
            loss = self.objective(predicted_encs, actions_transposed)  # (1,)

            total_loss = loss.mean()
            total_loss.backward()

            # Manual gradient descent update with noise
            with torch.no_grad():
                actions_new = actions - self.lr * actions.grad

                # Add Gaussian noise if specified
                if self.action_noise > 0:
                    actions_new += torch.randn_like(actions_new) * self.action_noise

                # Apply clipping if max_norms is specified (similar to CEM)
                if self.max_norms is not None:
                    for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                        actions_new[:, :, dims] = torch.clip(actions_new[:, :, dims], min=-maxnorm, max=maxnorm)

                actions.copy_(actions_new)

            # Reset gradients after manual update
            actions.grad.zero_()

            losses.append(total_loss.item())

            # Store predictions for this iteration
            with torch.no_grad():
                predicted_best_encs = self.unroll(z_init, act_suffix=actions.transpose(0, 1))
                predicted_best_encs_over_iterations.append(predicted_best_encs)

                if self.decode_each_iteration and self.decode_unroll is not None:
                    pred_frames = self.decode_unroll(predicted_best_encs)
                    pred_frames_over_iterations.append(pred_frames)

        # Return the optimized actions
        final_actions = actions.squeeze(0).detach()
        losses = torch.tensor(losses).detach().unsqueeze(-1)

        result = PlanningResult(
            actions=final_actions[: self.num_act_stepped] if self.num_act_stepped else final_actions,
            losses=losses,
            prev_elite_losses_mean=losses,
            prev_elite_losses_std=torch.zeros_like(losses),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class AdamPlanner(GradientDescentPlanner):
    """Adam optimizer-based planner for action optimization in latent space.

    This is a convenience wrapper around GradientDescentPlanner with optimizer_type="adam".
    """

    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int,
        iterations: int = 500,
        lr: float = 1,
        action_noise: float = 0.003,
        sample_type: str = "randn",
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        adam_betas: tuple = (0.9, 0.995),
        adam_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(
            unroll=unroll,
            action_dim=action_dim,
            horizon=horizon,
            iterations=iterations,
            lr=lr,
            action_noise=action_noise,
            sample_type=sample_type,
            var_scale=var_scale,
            max_norms=max_norms,
            max_norm_dims=max_norm_dims,
            num_act_stepped=num_act_stepped,
            decode_each_iteration=decode_each_iteration,
            decode_unroll=decode_unroll,
            optimizer_type="adam",
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            **kwargs,
        )


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
    









































# ----------------------------------------------------
























# from https://github.com/facebookresearch/eb_jepa/blob/main/eb_jepa/planning.py







import os
import time
from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.vis_utils import (
    analyze_distances,
    create_comparison_gif,
    plot_losses,
    save_decoded_frames,
    save_gif,
    show_images,
)

logger = get_logger(__name__)

planner_name_map = {
    "cem": "CEMPlanner",
    "mppi": "MPPIPlanner",
}
objective_name_map = {
    "repr_dist": "ReprTargetDistMPCObjective",
}


def main_unroll_eval(
    model,
    env_creator,
    eval_folder,
    num_samples=4,
    loader=None,
    prober=None,
    cfg=None,
):
    """
    Evaluate the model's unrolling capabilities by comparing unrolled predictions to ground truth.
    """
    env = env_creator()
    env.reset()
    device = next(model.parameters()).device
    normalizer = (
        loader.dataset.normalizer if hasattr(loader.dataset, "normalizer") else None
    )
    agent = GCAgent(
        model=model, plan_cfg=None, normalizer=normalizer, env=env, loc_prober=prober
    )
    mse_values = []
    position_mse_values = []
    unroll_times = []
    loader_iter = iter(loader)

    for idx in tqdm(
        range(num_samples), desc="Evaluating unroll", disable=cfg.logging.tqdm_silent
    ):
        try:
            x, a, loc, wall_x, door_y = next(loader_iter)
        except StopIteration:
            logger.warning(
                f"Loader exhausted after {idx} samples (requested {num_samples})"
            )
            break

        x = x.to(device)
        a = a.to(device)
        with torch.no_grad():
            obs_init = x[:, :, 0:1]  # B C T H W
            start_time = time.time()
            predicted_states = agent.unroll(obs_init, a, repeat_batch=False)[
                :, :, :-1
            ]  # discard last predicted state
            end_time = time.time()
            unroll_times.append(end_time - start_time)
            rand_predicted_states = agent.unroll(
                obs_init, torch.randn_like(a), repeat_batch=False
            )[
                :, :, :-1
            ]  # B D T H W
            # To ensure independence across timesteps when encoding the sequence, batchify it
            # There is no independence between timesteps when using GroupNorm, even in eval mode
            B, C, T, H, W = x.shape
            gt_encoded = (
                model.encode(x.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2))
                .squeeze(2)
                .unflatten(dim=0, sizes=(B, -1))
                .permute(0, 2, 1, 3, 4)
            )
            latent_mse = (
                ((gt_encoded - predicted_states) ** 2).mean(dim=(1, 3, 4)).cpu().numpy()
            )  # B T
            mse_values.append(latent_mse)

            if prober:
                gt_decoded = agent.decode_loc_to_pixel(gt_encoded, wall_x, door_y)
                pred_decoded = agent.decode_loc_to_pixel(
                    predicted_states, wall_x, door_y
                )
                rand_pred_decoded = agent.decode_loc_to_pixel(
                    rand_predicted_states, wall_x, door_y
                )  # B T H W C
                gt_frames = agent.normalizer.unnormalize_state(
                    x.permute(0, 2, 1, 3, 4)
                ).permute(0, 1, 3, 4, 2)
                gt_frames = (
                    (gt_frames * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                )  # B T H W C uint8

                # Decode positions from predicted_states and compute MSE with ground truth
                B_probe, D_probe, T_probe, H_probe, W_probe = predicted_states.shape
                pred_positions = (
                    prober.apply_head(predicted_states).permute(0, 2, 1).cpu()
                )  # B T 2
                gt_positions = loc.permute(0, 2, 1)  # B T 2
                position_mse = (
                    ((pred_positions - gt_positions.cpu()) ** 2)
                    .mean(dim=-1)
                    .cpu()
                    .numpy()
                )  # B T
                position_mse_values.append(position_mse)

                create_comparison_gif(
                    gt_frames,
                    pred_decoded,
                    rand_pred_decoded,
                    gt_dec=gt_decoded,
                    save_path=f"{eval_folder}/b{idx}.gif",
                )
    all_mse_values = np.vstack(mse_values)  # Shape: [num_batches, T]
    mean_mse_per_timestep = np.mean(all_mse_values, axis=0)  # Shape: [T]
    std_mse_per_timestep = np.std(all_mse_values, axis=0)  # Shape: [T]
    avg_unroll_time = np.mean(unroll_times)
    results = {}
    for t in range(mean_mse_per_timestep.shape[0]):
        results[f"val_rollout/mean_mse/{t}"] = mean_mse_per_timestep[t]
        results[f"val_rollout/std_mse/{t}"] = std_mse_per_timestep[t]

    # Log position MSE if prober was used
    if len(position_mse_values) > 0:
        all_position_mse_values = np.vstack(
            position_mse_values
        )  # Shape: [num_batches, T]
        mean_position_mse_per_timestep = np.mean(
            all_position_mse_values, axis=0
        )  # Shape: [T]
        std_position_mse_per_timestep = np.std(
            all_position_mse_values, axis=0
        )  # Shape: [T]
        for t in range(mean_position_mse_per_timestep.shape[0]):
            results[f"val_rollout/mean_pos_mse/{t}"] = mean_position_mse_per_timestep[t]
            results[f"val_rollout/std_pos_mse/{t}"] = std_position_mse_per_timestep[t]

    results["avg_unroll_time"] = avg_unroll_time

    pd.DataFrame([results]).to_csv(f"{eval_folder}/eval.csv", index=None)
    return results


### Main planning eval loop ###
def main_eval(
    plan_cfg,
    model,
    env_creator,
    eval_folder,
    num_episodes=10,
    loader=None,
    prober=None,
):
    plan_cfg = OmegaConf.create(plan_cfg)
    env = env_creator()
    env.reset()

    agent = GCAgent(
        model,
        action_dim=2,
        plan_cfg=plan_cfg,
        normalizer=env.normalizer,
        loc_prober=prober,
        env=env,
    )
    logger.info(f"Agent created with planner {agent.planner.__class__.__name__}")
    logger.info(f"Planning with {plan_cfg=}")

    successes = []
    distances = []
    episode_times = []
    episode_observations = []
    episode_infos = []

    for ep in range(num_episodes):
        episode_start_time = time.time()
        ep_folder = eval_folder / f"ep_{ep}"
        os.makedirs(ep_folder, exist_ok=True)
        if agent.decode_each_iteration:
            ep_plan_vis_dir = ep_folder / "plan_vis"
            os.makedirs(ep_plan_vis_dir, exist_ok=True)

        if plan_cfg.task_specification.goal_source == "dset":
            obs_slice, a, loc, _, _ = next(iter(loader))
            # obs, init_loc = obs_slice[0], loc[0]
            # goal_img, goal_loc = obs_slice[-1], loc[-1]  # [C, H, W] uint8 tensor
            # env.set_goal(goal_img) # Set goal in the environment
        elif plan_cfg.task_specification.goal_source == "random_state":
            obs, info = env.reset()  # [C, H, W] uint8 tensor
            obs, reward, done, truncated, info = env.step(
                np.zeros(env.action_space.shape[0])
            )  # step with zero action to get the first observation
            goal_img = info["target_obs"]  # [C, H, W] uint8 tensor

        combined = torch.stack([obs, goal_img], dim=0)
        show_images(
            combined,
            nrow=2,  # Both images in one row
            titles=["Init", "Goal"],
            save_path=f"{ep_folder}/state.pdf",
            close_fig=True,
            first_channel_only=False,
            clamp=False,
        )
        agent.set_goal(
            goal_img.detach().clone().to(dtype=torch.float32),
            info["target_position"],
        )

        done = False
        steps_left = env.n_allowed_steps
        pbar = tqdm(
            desc="executing agent",
            total=steps_left,
            leave=True,
            disable=plan_cfg.logging.tqdm_silent,
        )
        t0 = True

        observations = [obs]
        infos = [info]

        prev_losses = []
        prev_elite_losses_mean = []
        prev_elite_losses_std = []

        while steps_left > 0:
            # while (not done and steps_left > 0):
            plan_vis_path = (
                f"{ep_plan_vis_dir}/step{env.n_allowed_steps - steps_left}"
                if agent.decode_each_iteration
                else None
            )
            # first loop iter: obs is from reset(), then it is from step()
            obs_tensor = (
                env.normalizer.normalize_state(
                    obs.detach().clone().to(dtype=torch.float32, device=agent.device)
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )  # Unsqueeze the batch and time dimensions : C H W -> 1 C 1 H W
            with torch.no_grad():
                action = (
                    agent.act(
                        obs_tensor,
                        steps_left=steps_left,
                        t0=t0,
                        plan_vis_path=plan_vis_path,
                    )
                    .cpu()
                    .numpy()
                )  # T, A
            if agent._prev_losses is not None:
                prev_losses.append(agent._prev_losses)
                prev_elite_losses_mean.append(agent._prev_elite_losses_mean)
                prev_elite_losses_std.append(agent._prev_elite_losses_std)
            for a in action:
                obs, reward, done, truncated, info = env.step(a)
                t0 = False
                observations.append(obs)
                infos.append(info)
                steps_left -= 1
                pbar.update(1)
                eval_results = env.eval_state(
                    info["target_position"], info["dot_position"]
                )
                success = eval_results["success"]
                state_dist = eval_results["state_dist"]
            pbar.set_postfix({"success": success, "state_dist": state_dist})
        pbar.close()

        episode_observations.append(torch.stack(observations))
        episode_infos.append(infos)
        successes.append(success)
        distances.append(state_dist)

        if plan_cfg.logging.get("optional_plots", True):
            analyze_distances(
                episode_observations[-1],
                episode_infos[-1],
                str(ep_folder / "agent"),
                goal_position=agent.goal_position,
                goal_state=agent.goal_state,
                normalizer=agent.normalizer,
                model=agent.model,
                objective=agent.objective,
                device=agent.device,
            )
            plot_losses(
                prev_losses,
                prev_elite_losses_mean,
                prev_elite_losses_std,
                work_dir=ep_folder,
                num_act_stepped=agent.num_act_stepped,
            )
            save_path = f"{ep_folder}/agent_steps_{'succ' if success else 'fail'}.gif"
        save_gif(
            episode_observations[-1],
            save_path=save_path,
            show_frame_numbers=True,
            fps=20,
            init_frame=observations[0],
            goal_frame=goal_img,
        )
        logger.info(f"GIF saved to {save_path}")
        episode_end_time = time.time()  # Add this line
        episode_times.append(episode_end_time - episode_start_time)
    avg_episode_time = np.mean(episode_times)
    task_data = {
        "success_rate": np.mean(successes),
        "mean_state_dist": np.mean(distances),
        "avg_episode_time": avg_episode_time,
    }
    pd.DataFrame([task_data]).to_csv(f"{eval_folder}/eval.csv", mode="a", index=None)
    return task_data


### Goal-conditioned agent for planning ###
class GCAgent:
    def __init__(
        self,
        model,
        action_dim=2,
        plan_cfg=None,
        normalizer: Optional[Callable] = None,
        loc_prober: Optional[Callable] = None,
        img_prober: Optional[Callable] = None,
        env: Optional[Callable] = None,
    ):
        self.plan_cfg = plan_cfg
        self.env = env
        self.model = model
        self.device = next(model.parameters()).device
        self.loc_prober = loc_prober
        self.img_prober = img_prober
        self.normalizer = normalizer

        # Set default values if plan_cfg is None
        if plan_cfg is None:
            self.decode_each_iteration = False
            self.num_act_stepped = 1
            self.planner = None
            logger.info("No plan_cfg provided in GCAgent, planner not initialized.")
        else:
            self.decode_each_iteration = plan_cfg.planner.get(
                "decode_each_iteration", False
            )
            self.num_act_stepped = plan_cfg.planner.get("num_act_stepped", 1)
            planner_name = plan_cfg.planner.get("planner_name", "cem")
            planner_class_name = planner_name_map[planner_name]
            planner_class = globals()[planner_class_name]
            if planner_class is not None:
                self.planner = planner_class(
                    unroll=self.unroll,
                    action_dim=action_dim,
                    decode_loc_to_pixel=self.decode_loc_to_pixel,
                    **plan_cfg.planner,
                )
            else:
                logger.info("No planner provided in GCAgent.")
                self.planner = None

        self.goal_state = None
        self.goal_position = None
        self.goal_state_enc = None
        self._prev_losses = None

    def set_goal(self, goal_state, goal_position=None):
        self.goal_position = goal_position
        self.goal_state = goal_state
        # Unsqueeze the batch and time dimensions : C H W -> 1 C 1 H W
        self.goal_state_enc = self.model.encode(
            self.normalizer.normalize_state(goal_state.to(self.device))
            .unsqueeze(0)
            .unsqueeze(2)
        )
        objective_name = self.plan_cfg.planner.planning_objective.get(
            "objective_type", "repr_target_dist"
        )
        objective_class_name = objective_name_map[objective_name]
        objective_class = globals()[objective_class_name]
        self.objective = objective_class(
            target_enc=self.goal_state_enc, **self.plan_cfg.planner.planning_objective
        )
        self.planner.set_objective(self.objective)

    def unroll(self, obs_init, actions, repeat_batch=True):
        """
        Unroll the model for planning.

        Args:
            obs_init: [B, C, T, H, W]
            actions: [B, A, T]

        Returns:
            predicted_states: [B, D, T, H, W]
        """
        batch_size = actions.shape[0]
        nsteps = actions.shape[2]
        if repeat_batch:
            obs_init = obs_init.repeat(batch_size, 1, 1, 1, 1)
        predicted_states, _ = self.model.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=self.plan_cfg["ctxt_window_time"] if self.plan_cfg else 1,
            compute_loss=False,
            return_all_steps=False,
        )
        return predicted_states

    def decode_loc_to_pixel(self, predicted_encs, wall_x=None, door_y=None):
        """
        Decode the predicted encodings into frames.

        Args:
            predicted_encs: [B, D, T, H, W]

        Returns:
            np.array of shape [B, T, H, W, C] on cpu for visualization.
        """
        assert self.loc_prober is not None
        B, D, T, H, W = predicted_encs.shape
        out = self.loc_prober.apply_head(predicted_encs).permute(0, 2, 1).cpu()  # B T 2
        out = self.normalizer.unnormalize_location(out)  # B T 2
        frames = self.env.coord_to_pixel(out, wall_x=wall_x, door_y=door_y)  # B T C H W
        frames = frames.permute(0, 1, 3, 4, 2).cpu().numpy()  # B T H W C
        return frames

    def act(self, obs, steps_left=None, t0=False, plan_vis_path=None):
        planning_result = self.planner.plan(
            obs,
            steps_left=steps_left,
            eval_mode=True,
            t0=t0,
            plan_vis_path=plan_vis_path,
        )
        self._prev_losses = planning_result.losses
        self._prev_elite_losses_mean = planning_result.prev_elite_losses_mean
        self._prev_elite_losses_std = planning_result.prev_elite_losses_std
        return planning_result.actions[: self.num_act_stepped]  # T, A


### Planning objectives to minimize ###
class ReprTargetDistMPCObjective:
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        target_enc: torch.Tensor,
        sum_all_diffs: bool = False,
        **kwargs,
    ):
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        """
        Args:
            encodings: [B, D, T, H, W]
            keepdims: if True, return [B, T], else return [B]

        Returns:
            diff: [B, T] else [B] if sum_all_diffs or not keepdims
        """
        if self.sum_all_diffs:
            keepdims = True
        target = self.target_enc
        if target.shape != encodings.shape:
            target = target.expand(encodings.shape[0], -1, encodings.shape[2], -1, -1)

        metric = torch.nn.MSELoss(reduction="none")
        diff = metric(target, encodings).mean(dim=(1, 3, 4))  # B T
        if not keepdims:
            diff = diff[:, -1]
        if self.sum_all_diffs:
            diff = diff.sum(dim=1)
        return diff


### Planning optimizers interface ###
class PlanningResult(NamedTuple):
    actions: torch.Tensor
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None


class Planner(ABC):
    def __init__(self, unroll: Callable, **kwargs):
        self.unroll = unroll
        self.objective = None

    def set_objective(self, objective: Callable):
        self.objective = objective

    @abstractmethod
    def plan(
        self,
        obs_init: torch.Tensor,
        steps_left: Optional[int] = None,
        t0: bool = False,
        eval_mode: bool = False,
    ):
        pass

    def cost_function(
        self, actions: torch.Tensor, obs_init: torch.Tensor
    ) -> torch.Tensor:
        predicted_encs = self.unroll(obs_init, actions)
        return self.objective(predicted_encs)


### Specific planning optimizers ###
class CEMPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 30,
        num_samples: int = 300,
        plan_length: int = 15,
        action_dim: int = 2,
        var_scale: float = 1,
        num_elites: int = 10,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = True,
        decode_loc_to_pixel: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_loc_to_pixel = decode_loc_to_pixel

    @torch.no_grad()
    def plan(
        self, obs_init, steps_left=None, eval_mode=True, t0=False, plan_vis_path=None
    ):
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        # Initialize mean and std for the action distribution
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(
            plan_length, self.action_dim, device=self.device
        )

        # Initialize actions tensor
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses = []
        elite_means = []
        elite_stds = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # CEM iterations
        for _ in range(self.n_iters):
            # Sample actions
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )  # T B A

            # Apply clipping if max_norms is specified
            if self.max_norms is not None:
                assert len(self.max_norms) == 1
                max_norm = self.max_norms[0]
                eps = 1e-6
                norms = actions.norm(dim=-1, keepdim=True)
                # calculate min and max allowed step sizes
                max_norms = torch.ones_like(norms) * max_norm
                min_norms = torch.ones_like(norms) * 0
                coeff = torch.min(torch.max(norms, min_norms), max_norms) / (
                    norms + eps
                )
                actions = actions * coeff

            # Compute costs
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Update parameters
            mean = torch.mean(elite_actions, dim=1)
            std = torch.std(elite_actions, dim=1)

            if self.decode_each_iteration:
                predicted_best_encs = self.unroll(
                    obs_init, rearrange(mean, "t a -> 1 a t")
                )
                pred_frames = self.decode_loc_to_pixel(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames.squeeze(0))
                # [T H W 3]: uint 8 in [0, 255]
        if self.decode_each_iteration:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)

        # Return the first action(s)
        a = mean

        return PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )


class MPPIPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 15,
        num_samples: int = 500,
        plan_length: int = 15,
        action_dim: int = 2,
        max_std: float = 2,
        num_elites: int = 64,
        temperature: float = 0.005,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = False,
        decode_loc_to_pixel: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_std = max_std
        self.num_elites = num_elites
        self.temperature = temperature
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_loc_to_pixel = decode_loc_to_pixel
        self._prev_mean = None

    @torch.no_grad()
    def plan(
        self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None
    ):
        """
        Args:
                obs_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(
            plan_length, self.action_dim, device=self.device
        )
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses = []
        elite_means = []
        elite_stds = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []

        # MPPI iterations
        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )  # T B A
            # Compute costs
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Update parameters
            min_cost = cost.min(0)[0]
            score = torch.exp(
                self.temperature * (min_cost - elite_loss[:, 0])
            )  # increasing with elite_value
            score /= score.sum(0)
            mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
            ) / (  # T B A
                score.sum(0) + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,  # T B A
                )
                / (score.sum(0) + 1e-9)
            )
            if self.decode_each_iteration:
                predicted_best_encs = self.unroll(
                    obs_init, rearrange(mean, "t a -> 1 a t")
                )
                pred_frames = self.decode_loc_to_pixel(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames.squeeze(0))
                # [T H W 3]: uint 8 in [0, 255]
        if self.decode_each_iteration:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)
        # Select action
        score = score.cpu().numpy()
        actions = elite_actions[
            :, np.random.choice(np.arange(score.shape[0]), p=score)
        ]  # T, A
        self._prev_mean = mean
        if not eval_mode:
            actions += std * torch.randn(
                self.action_dim, device=std.device, generator=self.local_generator
            )

        return PlanningResult(
            actions=actions,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )











# ------------------------------------------------------------











# from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py








import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)