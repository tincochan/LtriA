import sys
from gym_partially_observable_grid.envs import PartiallyObservableWorld

# Get Key by given value for a dictionary
def getKeyByValue(dict, value):
    return {i for i in dict if dict[i] == value}.pop()

class world2prismParser():
    def __init__(self, input_file):
        self.input_file = input_file





    # Calculate next location by given location and action
    def getNextLocation(self, location, action):
        (x, y) = location
        if action == 'up' and x > self.x_min:
            x -= 1
        elif action == 'down' and x < self.x_max:
            x += 1
        elif action == 'left' and y > self.y_min:
            y -= 1
        elif action == 'right' and y < self.y_max:
            y += 1
        else:
            return None
        if (x, y) in self.not_accessible_locations:
            return None
        return getKeyByValue(self.location_dict, (x, y))


    # Get terminal locations from abstract world
    def getTerminalLocations(self):
        abstract_world = self.parsed_world.abstract_world
        terminal_locations = []
        for i in range(1, self.x_max + 1):
            for j in range(1, self.y_max + 1):
                if abstract_world[i][j] == 'd':
                    terminal_locations.append((i, j))
        return terminal_locations


    # Get wall locations from parsed world
    def getWallLocations(self):
        world = self.parsed_world.world
        wall_locations = []
        # loop without border
        for i in range(2, self.x_max):
            for j in range(2, self.y_max):
                if world[i][j] == '#':
                    wall_locations.append((i, j))
        return wall_locations


    # Handle stochastic tite for a given location
    def handle_stochastic_tile(self, stochastic_tile_location):
        tmp_string = ''
        stochastic_tile = self.parsed_world.stochastic_tile.get(stochastic_tile_location)
        # action_behaviour = parsed_world.rules.items().mapping.get(str(stochastic_tile)).behaviour
        action_behaviour = self.parsed_world.rules[str(stochastic_tile)].behaviour
        location_index = getKeyByValue(self.location_dict, stochastic_tile_location)
        for expected_action in self.parsed_world.actions_dict.keys():
            tmp_string += f'[{expected_action}] loc={location_index} ->\n\n'
            stochastic_tile_action_behaviour = action_behaviour.get(self.parsed_world.actions_dict.get(expected_action))
            for (action, prob) in stochastic_tile_action_behaviour:
                action_name = getKeyByValue(self.parsed_world.actions_dict, action)
                next_loc = self.getNextLocation(stochastic_tile_location, action_name)
                if next_loc is None:
                    next_loc = location_index
                tmp_string += f' {prob} : (loc\'={next_loc}) + '
            tmp_string = tmp_string[:-3] + ';\n\n'
        return tmp_string


    def parse_world(self):
        self.parsed_world = PartiallyObservableWorld(self.input_file, False, True, False, True)
        self.x_min = 1
        self.y_min = 1
        self.x_max = len(self.parsed_world.world) - 2
        self.y_max = len(self.parsed_world.world[0]) - 2
        # Initialize location dictionary by indices and locations (x, y)
        self.location_dict = {}
        location_index = 0
        # leave out the outer border
        for i in range(1, self.x_max + 1):
            for j in range(1, self.y_max + 1):
                self.location_dict[location_index] = (i, j)
                location_index += 1
        location_index -= 1
        initial_index = getKeyByValue(self.location_dict, self.parsed_world.initial_location)
            # Initialize temp string for MDP
        tmp_module = ''
        tmp_module += f'loc : [0..{location_index}] init {initial_index};\n\n'

        # Define not-accessible locations and stochastic tile locations
        self.not_accessible_locations = self.getWallLocations()
        self.stochastic_tile_locations = self.parsed_world.stochastic_tile

        # Go through all locations in the dictionary and check:
        # - if the location is not-accessible -> leave out
        # - if the location is a stochastic tile location -> handle stochastic tile
        # - else: handle all possible actions and write it to the temp MDP string
        for loc_index in self.location_dict:
            location = self.location_dict.get(loc_index)
            if location not in self.not_accessible_locations:
                if location in self.stochastic_tile_locations:
                    tmp_module += self.handle_stochastic_tile(location)
                else:
                    for action in self.parsed_world.actions_dict.keys():
                        next_loc = self.getNextLocation(location, action)
                        if next_loc is not None:
                            tmp_module += f'[{action}] loc={loc_index} -> \n\n'
                            tmp_module += f' 1.0 : (loc\'={next_loc});\n\n'
                        else:
                            tmp_module += f'[{action}] loc={loc_index} -> \n\n'
                            tmp_module += f' 1.0 : (loc\'={loc_index});\n\n'

        # Create MDP string
        # modified game header because right now tempest shields
        # only work with stochastic multiplayer games
        
        game_header = """
            smg
            player robot
            [up], [down], [left], [right]
            endplayer
            player none
            none
            endplayer
            module none
            endmodule """

        mdp = ''
        # mdp += f'mdp\n\n'
        mdp += f'{game_header}\n\n'
        

        mdp += 'module moves\n\n'

        # Add parsed actions
        mdp += tmp_module
        mdp += 'endmodule\n\n'

        # Add labels:
        # -Init:
        mdp += f'label \"Init\" = loc={initial_index};\n\n'

        # -Termination:
        termination_locations = self.getTerminalLocations()
        if len(termination_locations) < 1:
            print('[ERROR] Invalid World: World must have at least 1 position with label \"death\" ')
            sys.exit(2)
        mdp += 'label \"death\" = '
        for termination_location in termination_locations:
            termination_index = getKeyByValue(self.location_dict, termination_location)
            mdp += f'loc={termination_index}|'
        mdp = mdp[:-1] + ';\n\n'

        # -Tile:
        if len(self.stochastic_tile_locations) > 0:
            mdp += 'label \"tile\" = '
            for stochastic_tile_location in self.stochastic_tile_locations:
                stochastic_tile_index = getKeyByValue(self.location_dict, stochastic_tile_location)
                mdp += f'loc={stochastic_tile_index}|'
            mdp = mdp[:-1] + ';\n\n'

        # TODO: get/add another labels?

        # -Goal:
        if len(self.parsed_world.goal_locations) < 1:
            print('[ERROR] Invalid World: World must have at least 1 position with label \"GOAL\" ')
            sys.exit(2)
        mdp += 'label \"GOAL\" = '
        for goal_location in self.parsed_world.goal_locations:
            goal_index = getKeyByValue(self.location_dict, goal_location)
            mdp += f'loc={goal_index}|'
        mdp = mdp[:-1] + ';\n\n'

        # Rewards:
        if len(self.parsed_world.reward_tiles) > 0:
            mdp += 'rewards\n'
            for reward_location in self.parsed_world.reward_tiles:
                reward_location_index = getKeyByValue(self.location_dict, reward_location)
                reward = self.parsed_world.reward_tiles.get(reward_location)
                mdp += f'[] loc={reward_location_index} : {reward};\n'
            mdp += 'endrewards\n'

        # Print MDP
        # print(mdp)

        # Write MDP to file
        # file = open(input_file + '.prism', 'w')
        file = open('test_worlds/po_rl.prism', 'w')
        file.write(mdp)
        file.close()


def main(input_file):

    # Initialize parsed world from input file by using PartiallyObservableWorld
    worldprismparser = world2prismParser(input_file=input_file)
    worldprismparser.parse_world()    



if __name__=="__main__":
    # check arguments
    if len(sys.argv) != 2:
        print('[ERROR] Usage: parse_world.py <input_file>')
        sys.exit(1)
    else:
        input_file = sys.argv[1]
    main(input_file)
