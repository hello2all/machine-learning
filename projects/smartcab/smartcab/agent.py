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
        self.Qlearner = Qlearn(self.actions)
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

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
        state_string  = ''
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
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=10000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    pickle.dump(a.Qlearner.q, open('q.pkl','wb'))

if __name__ == '__main__':
    run()
