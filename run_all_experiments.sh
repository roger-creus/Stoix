

#PQN
### Breakout
python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=1
python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=2
python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=3

### Space Invaders
python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=1
python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=2
python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=3

# PQN - Dueling
### Breakout
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_dueling_dqn system.system_name=pqn_dueling network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=1
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_dueling_dqn system.system_name=pqn_dueling network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=2
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_dueling_dqn system.system_name=pqn_dueling network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=3

### Space Invaders
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_dueling_dqn system.system_name=pqn_dueling network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=1
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_dueling_dqn system.system_name=pqn_dueling network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=2
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_dueling_dqn system.system_name=pqn_dueling network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=3

# PQN - Noisy
### Breakout
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_noisy_dqn system.system_name=pqn_noisy network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=1
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_noisy_dqn system.system_name=pqn_noisy network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=2
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_noisy_dqn system.system_name=pqn_noisy network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/breakout arch.seed=3

### Space Invaders
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_noisy_dqn system.system_name=pqn_noisy network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=1
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_noisy_dqn system.system_name=pqn_noisy network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=2
python stoix/systems/pqn/anakin/ff_pqn.py network=mlp_noisy_dqn system.system_name=pqn_noisy network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/space_invaders arch.seed=3

# ### Freeway
# python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/freeway arch.seed=1
# python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/freeway arch.seed=2
# python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/freeway arch.seed=3

# ### Asterix
# python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/asterix arch.seed=1
# python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/asterix arch.seed=2
# python stoix/systems/pqn/anakin/ff_pqn.py network.actor_network.pre_torso.use_layer_norm=True logger.use_json=True env=gymnax/asterix arch.seed=3