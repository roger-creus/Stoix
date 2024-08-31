#Minatar/Breakout

python stoix/systems/pqn/anakin/ff_pqn.py env=gymnax/breakout arch.seed=1 network.actor_network.pre_torso.use_layer_norm=True
python stoix/systems/pqn/anakin/ff_pqn.py env=gymnax/breakout arch.seed=2 network.actor_network.pre_torso.use_layer_norm=True
python stoix/systems/pqn/anakin/ff_pqn.py env=gymnax/breakout arch.seed=3 network.actor_network.pre_torso.use_layer_norm=True

# python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/breakout arch.seed=1
# python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/breakout arch.seed=2
# python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/breakout arch.seed=3

# python stoix/systems/mpo/ff_vmpo.py env=gymnax/breakout arch.seed=1
# python stoix/systems/mpo/ff_vmpo.py env=gymnax/breakout arch.seed=2
# python stoix/systems/mpo/ff_vmpo.py env=gymnax/breakout arch.seed=3

#Minatar/SpaceInvaders

python stoix/systems/pqn/anakin/ff_pqn.py env=gymnax/space_invaders arch.seed=1 network.actor_network.pre_torso.use_layer_norm=True
python stoix/systems/pqn/anakin/ff_pqn.py env=gymnax/space_invaders arch.seed=2 network.actor_network.pre_torso.use_layer_norm=True
python stoix/systems/pqn/anakin/ff_pqn.py env=gymnax/space_invaders arch.seed=3 network.actor_network.pre_torso.use_layer_norm=True

# python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/space_invaders arch.seed=1
# python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/space_invaders arch.seed=2
# python stoix/systems/ppo/anakin/ff_ppo.py env=gymnax/space_invaders arch.seed=3

# python stoix/systems/mpo/ff_vmpo.py env=gymnax/space_invaders arch.seed=1
# python stoix/systems/mpo/ff_vmpo.py env=gymnax/space_invaders arch.seed=2
# python stoix/systems/mpo/ff_vmpo.py env=gymnax/space_invaders arch.seed=3