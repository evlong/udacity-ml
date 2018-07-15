import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        self.init_pose = init_pose
        self.init_velocities = init_velocities
        self.init_angle_velocities = init_angle_velocities
        # Simulation
        
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3 #for each timestep of the agent, step the simulation 3 timesteps, repeating the agent's action and rendering each time.

        #self.state_size = self.action_repeat * 9
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal - to land! 
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.2*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 1 if reward >= 1 else reward
        reward = -1 if reward <-1 else reward
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            state_all.append(self.sim.pose)
            #state_all.append(np.concatenate((self.sim.pose,self.sim.v),))
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate(([self.sim.pose]+[self.sim.v]) * self.action_repeat) 
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state