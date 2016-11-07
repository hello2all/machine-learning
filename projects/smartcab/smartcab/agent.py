import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from Qlearn import Qlearn
import pickle

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # Initialize any additional variables here
        self.actions = valid_actions = [None, 'forward', 'left', 'right']
        self.state = None
        self.Qlearner = Qlearn(self.actions) # train
        # test
        # self.Qlearner = Qlearn(self.actions, epsilon=0.0)
        # self.Qlearner.q = pickle.load(open("q.pkl"))
    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0

        return None

    def vectorize(self, value): # return a index for possible values, 0,1,2,4 for 'None', 'forward', 'left', 'right' respectively and 0,1 for 'green', 'red'respectively
        if value == None or value == 'green':
            return '0'
        if value == 'forward' or value == 'red':
            return '1'
        if value == 'left':
            return '2'
        if value == 'right':
            return '3'
    def sense(self, inputs, next_waypoint):

        self.state = self.vectorize(inputs['light']) + self.vectorize(inputs['oncoming']) + self.vectorize(inputs['right']) + self.vectorize(inputs['left']) + self.vectorize(next_waypoint)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.sense(inputs, self.next_waypoint)
        prev_state = self.state
        # TODO: Select action according to your policy
        action = self.Qlearner.chooseAction(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Gather inputs again
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.sense(inputs, self.next_waypoint)

        # TODO: Learn policy based on state, action, reward
        self.Qlearner.learn(prev_state, action, reward, self.state)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        # print self.state

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run()


if __name__ == '__main__':
    run()
