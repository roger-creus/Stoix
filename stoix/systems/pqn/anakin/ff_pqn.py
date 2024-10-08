import copy
import time
from typing import Any, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    CriticApply,
    AnakinExperimentOutput,
    LearnerFn,
    OnPolicyLearnerState,
)
from stoix.networks.base import FeedForwardActor as Actor
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.systems.pqn.pqn_types import PQNTransition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.loss import pqn_learning
from stoix.utils.multistep import batch_lambda_returns
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: Environment,
    q_apply_fn: Tuple[ActorApply, CriticApply],
    q_update_fn: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[OnPolicyLearnerState]:
    """Get the learner function."""

    def _update_step(
        learner_state: OnPolicyLearnerState, _: Any
    ) -> Tuple[OnPolicyLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: OnPolicyLearnerState, _: Any
        ) -> Tuple[OnPolicyLearnerState, PQNTransition]:
            """Step the environment."""
            q_params, opt_states, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = q_apply_fn(q_params, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)
            max_q_value = actor_policy.preferences.max(-1)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = PQNTransition(
                done,
                truncated,
                action,
                max_q_value,
                timestep.reward,
                last_timestep.observation,
                info,
            )
            learner_state = OnPolicyLearnerState(q_params, opt_states, key, env_state, timestep)
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE Q(lambda) TARGETS
        q_params, opt_states, key, env_state, last_timestep = learner_state

        # TODO: is this really not necessary?
        #last_val = q_apply_fn(q_params, last_timestep.observation).preferences.max(-1)
        #v_t = jnp.concatenate([traj_batch.value, last_val[None, ...]], axis=0)
        r_t = traj_batch.reward
        v_t = traj_batch.value
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        
        targets = batch_lambda_returns(
            r_t,
            d_t,
            v_t,
            config.system.q_lambda,
            stop_target_gradients=True, # TODO: this should be true right?
            time_major=True,
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                q_params, opt_states = train_state
                traj_batch, targets = batch_info

                def _q_loss_fn(
                    q_params: FrozenDict,
                    traj_batch: PQNTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    
                    value = q_apply_fn(q_params, traj_batch.obs).preferences

                    value_loss = pqn_learning(
                        value, targets, traj_batch.action, huber_loss_parameter=0.0
                    )

                    loss_info = {
                        "value_loss": value_loss,
                    }
                    return value_loss, loss_info

                # CALCULATE CRITIC LOSS
                q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
                q_grads, loss_info = q_grad_fn(
                    q_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                q_grads, loss_info = jax.lax.pmean(
                    (q_grads, loss_info), axis_name="batch"
                )
                # pmean over devices.
                q_grads, loss_info = jax.lax.pmean(
                    (q_grads, loss_info), axis_name="device"
                )

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                q_updates, q_new_opt_state = q_update_fn(
                    q_grads, opt_states
                )
                
                q_new_params = optax.apply_updates(q_params, q_updates)

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = q_new_params
                new_opt_state = q_new_opt_state

                return (new_params, new_opt_state), loss_info

            q_params, opt_states, traj_batch, targets, key = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (q_params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (q_params, opt_states), minibatches
            )

            update_state = (q_params, opt_states, traj_batch, targets, key)
            return update_state, loss_info

        update_state = (q_params, opt_states, traj_batch, targets, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        q_params, opt_states, traj_batch, targets, key = update_state
        learner_state = OnPolicyLearnerState(q_params, opt_states, key, env_state, last_timestep)
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OnPolicyLearnerState,
    ) -> AnakinExperimentOutput[OnPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[OnPolicyLearnerState], Actor, OnPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    action_dim = int(env.action_spec().num_values)
    config.system.action_dim = action_dim

    # PRNG keys.
    key, q_net_key = keys

    # Define network and optimiser.
    q_network_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.training_epsilon,
    )
    
    q_network = Actor(torso=q_network_torso, action_head=q_network_action_head)
    
    eval_q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.evaluation_epsilon,
    )
    eval_q_network = Actor(torso=q_network_torso, action_head=eval_q_network_action_head)
    
    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.radam(q_lr),
    )

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise q params and optimiser state.
    q_online_params = q_network.init(q_net_key, init_x)
    q_opt_state = q_optim.init(q_online_params)

    # Pack params.
    params = q_online_params
    opt_states = q_opt_state

    q_network_apply_fn = q_network.apply

    # Pack apply and update functions.
    apply_fns = q_network_apply_fn
    update_fns = q_optim.update

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    replicate_learner = (params, opt_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = OnPolicyLearnerState(params, opt_states, step_keys, env_states, timesteps)

    return learn, eval_q_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, q_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=3
    )

    # Setup learner.
    learn, eval_q_network, learner_state = learner_setup(
        env, (key, q_net_key), config
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config, eval_q_network.apply),
        params=learner_state.params,
        config=config,
    )

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = jnp.float32(-1e7)
    best_params = unreplicate_batch_dim(learner_state.params)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params
        )  # Select only actor params
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../../configs/default/anakin",
    config_name="default_ff_pqn.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}PQN experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()