from ast import parse
import sys, argparse, os
import subprocess, json
from pathlib import Path
from tabnanny import verbose

import docker
from tqdm import tqdm
import random


import gym
import gym_partially_observable_grid
from parse_world import world2prismParser, getKeyByValue
import numpy as np
import pandas as pd

debug = False
docker_binary = 'docker'


class ShieldedQLearner():
    def __init__(self, world_file, params_file):
        self.world_file = world_file
        with open(params_file, 'r') as fp:
            self.params = json.load(fp)

        self.env = gym.make(id='poge-v1',
                world_file_path=self.world_file,
                force_determinism=False,
                indicate_slip=False,
                is_partially_obs=False,
                indicate_wall=False,
                one_time_rewards=True,
                step_penalty=self.params['step_penalty'])

        self.evaluation_df = pd.DataFrame(columns=['Episode', 'Goal', 'Death', 'EpLength', 'Reward', 'Return'])
        


    def parse_shield(self):
        with open('test_worlds/safety_shield.shield', 'r') as fp:
            data = fp.read().splitlines()

        # shield_dict[loc_value]['up'] = 0.798
        shield_dict = {}
        movenames = ['down', 'left', 'up', 'right']

        for line in data:
            if 'loc=' in line:
                loc_value = int(line.split('loc=')[1].split(']')[0])
                shield_dict[loc_value] = {}
                for movename in movenames:
                    movename_pos = line.find(movename)
                    if movename_pos != -1:
                        shield_dict[loc_value][movename] = float(line[:line[:movename_pos].rfind('(')-2].split(' ')[-1])
                    else:
                        shield_dict[loc_value][movename] = -1
        self.shield_dict = shield_dict


    def compute_shield(self):
        self.worldprismparser = world2prismParser(self.world_file)
        self.worldprismparser.parse_world()
        mount_to = "/tempest/examples"
        # mount_to = ""
        # prism_file = f"{mount_to}/slipery1.world.prism"
        # props_file = f"{mount_to}/propfile.props"
        tmp_dir = Path("test_worlds").absolute()
        threshold = self.params['threshold']
        prop = f'<safety_shield, PreSafety, lambda={threshold}> <<robot>> Pmax=? [ G !"death" ]'
        #open text file
        # proptext_file = open(f"{tmp_dir}/propfile.props", "w")
        proptext_file = open(f"{tmp_dir}/po_rl.prop", "w")

        # #write string to file
        proptext_file.write(prop)    
        # #close file
        proptext_file.close()

        client = docker.from_env()
        logos0 = client.containers.run("tempestshields/tempest:latest", f"tempest/run.sh", volumes = {tmp_dir : {'bind': mount_to, 'mode': 'rw'}}, stderr = True)
        # logos0 = client.containers.run("tempestshields/tempest:latest", f"tempest/build/bin/storm --prism {prism_file} --prop {props_file}", volumes = {tmp_dir : {'bind': mount_to, 'mode': 'rw'}}, stderr = True)
        print(logos0)
        

    def safe_argmax(self, state, debug = False):

        
        
        (x,y) = self.env.decode(state)
        loc_value = getKeyByValue(self.worldprismparser.location_dict, (x,y))
    
        best_q_value = -1e6
        best_action = -1
        for actionname, actionnum in self.env.actions_dict.items():
            if self.shield_dict[loc_value][actionname] > self.params['threshold']:
                q_value = self.q_table[state][actionnum]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = actionnum
        
        if debug:
            print(f"\n Starting with state {state}, (x,y) = {(x,y)}, loc={loc_value}, and the shield: {self.shield_dict[loc_value]}")
        if best_action == -1:
            if debug:
                print("No safe action found for safeargmax, proceeding with up.")
            best_action = 0
        return best_action
                

    def safe_random_sample(self, state):
        (x,y) = self.env.decode(state)
        loc_value = getKeyByValue(self.worldprismparser.location_dict, (x,y))
        actions_to_sample = []
        for actionname, actionnum in self.env.actions_dict.items():
            if self.shield_dict[loc_value][actionname] > self.params['threshold']:
                actions_to_sample.append(actionnum)
        if debug:
            print(f"\n Starting with state {state}, (x,y) = {(x,y)}, loc={loc_value}, and the shield: {self.shield_dict[loc_value]}")

        if len(actions_to_sample) == 0:
            if debug:
                print("No safe action found for saferandom, proceeding with up.")
            return 0
        return random.sample(actions_to_sample,1)[0]

        


    def get_abstract_output(self, state, reward):
        x, y = self.env.decode(state)
        if self.env.abstract_world[x][y] == 'd':
            output = 'death'
        elif reward == self.env.goal_reward:
            output = 'GOAL'
        else:
            output = self.env.abstract_world[x][y]
            if x + 1 < len(self.env.abstract_world) and self.env.abstract_world[x + 1][y] == 'd':
                output += 'death_r'
            if x - 1 >= 0 and self.env.abstract_world[x - 1][y] == 'd':
                output += 'death_l'
            if y + 1 < len(self.env.abstract_world[x]) and self.env.abstract_world[x][y + 1] == 'd':
                output += 'death_u'
            if y - 1 >= 0 and self.env.abstract_world[x][y - 1] == 'd':
                output += 'death_d'
        return output


    def train_agent(self, training_type='no_penalty',  verbose=False):
        interval_size = 100
        assert training_type in {'no_penalty', 'penalty', 'shielded'}
        verbose=False
        ep_reward = []
        training_forbidden = 0
        epsilon = self.params['epsilon'] # this initializes epsilon, then it will decay

        print(f'Training for {training_type} started.')

        training_data = []
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        for i in tqdm(range(1, self.params['number_of_training_episodes'] + 1)):
            # print(f"Episode {i}")
        
            epsilon = epsilon * self.params['epsilon_decay']
            if epsilon < self.params['epsilon_threshold']: 
                epsilon = self.params['epsilon_threshold']

            state = self.env.reset()

            episode_steps = [self.get_abstract_output(state, 0)]

            culmalative_reward = 0
            done = False
            while not done:
                if self.params['shield_training']:
                    if random.random() < epsilon:
                        action = self.safe_random_sample(state)
                    else:
                        action = self.safe_argmax(state)
                else:
                    if random.random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.q_table[state])

                next_state, reward, done, info = self.env.step(action)
                x, y = self.env.player_location[0], self.env.player_location[1]

                # If forbidden state is reached
                if self.env.abstract_world[x][y] == 'd':
                    training_forbidden += 1

                    reward = self.params['forbidden_state_reward']
                    done = True

                    # print(f"Episode: {episode_steps}, location: {env.player_location}")
                    # input("")

                old_value = self.q_table[state, action]

                next_max = np.max(self.q_table[next_state])

                new_value = (1 - self.params['alpha']) * old_value + self.params['alpha'] * (reward + self.params['gamma'] * next_max)
                self.q_table[state, action] = new_value

                culmalative_reward += reward
                if done:
                    ep_reward.append(culmalative_reward)
                    training_data.append(episode_steps)

                state = next_state
            if i % self.params['evaluate_every'] == 0:
                self.evaluate_agent(i)
        print("Training finished.")
        

    def evaluate_agent(self, episode = 0, shield=None, verbose = False):
        num_ep = self.params['number_of_evaluate_episodes']
        goals_reached = 0
        forbidden_state_reached = 0
        total_steps = 0
        total_reward=0
        total_return = 0
        total_deaths = 0
        accumulated_discount = 1



        for i in range(num_ep):
            state = self.env.reset()

            done = False
            accumulated_discount = 1
            while not done:
                if self.params['shield_evaluation']:
                    action = self.safe_argmax(state)
                else:
                    action = np.argmax(self.q_table[state])

                # if debug: print(f"[({env.player_location[0], env.player_location[1]}) [{state}]", end=": ")
                # if debug and shield:  print(f"allowed actions: {shield.get_safe_actions(state)}", end="")
                state, reward, done, info = self.env.step(action)
                # if debug: print(f"reward: {reward}]  ->", end="")

                x, y = self.env.player_location[0], self.env.player_location[1]
                if self.env.abstract_world[x][y] == 'd':
                    reward = self.params['forbidden_state_reward']
                    forbidden_state_reached += 1
                    done = True

                if reward == self.env.goal_reward and done:
                    goals_reached += 1

                total_steps += 1
                total_reward += reward
                total_return += accumulated_discount*reward
                accumulated_discount *= self.params['gamma']

            # if debug: input("")
        
        ev_results = [episode, goals_reached/num_ep, forbidden_state_reached/num_ep, total_steps/num_ep, total_reward/num_ep, total_return/num_ep]
        self.evaluation_df.loc[self.evaluation_df.shape[0],:] = ev_results
        if verbose:
            print(f"Results after {num_ep} episodes:")
            print(f"Total Number of Goal reached: {goals_reached} / {num_ep}")
            print(f"Average timesteps per episode: {total_steps / num_ep}")
            print(f"Average reward per episode: {total_reward / num_ep}")
            print(f"Average (discounted) return per episode: {total_return / num_ep}")


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--worldfile', type=str, default='test_worlds/slipery1.world',
                        help='File containing the world description.')
    parser.add_argument('--shield', type=bool, default=True,
                        help='Run agent shielded (True or False)')

    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Shield threshold, float between 0 and 1,')
    parser.add_argument('--params', type=str, default='params0.json',
                        help='Parameters file, json type.')
    args = parser.parse_args()
    print(args.threshold)



    QLearner = ShieldedQLearner(args.worldfile, args.params)


    # num_training_episodes = 30000 #12500 gets 0 reward, 13000 gets full reward
    if QLearner.params['shield_training'] or QLearner.params['shield_evaluation']:
        QLearner.compute_shield()
        QLearner.parse_shield()
    QLearner.train_agent()
    QLearner.evaluation_df.to_csv('evaluation_results.csv', index=False)
    print(QLearner.evaluation_df)
    QLearner.evaluation_df.plot(x='Episode', y='EpLength')
    QLearner.evaluation_df.plot(x='Episode', y='Return')
    # QLearner.evaluate_agent()


if __name__== "__main__":
    main()