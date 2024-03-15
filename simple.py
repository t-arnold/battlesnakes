# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
import time
import sys
import numpy as np
import math

random_seed = None

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "t-arnold",  
        "color": "#FFAA00",  
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    if random_seed is not None:
        random.seed(random_seed)
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")

def get_next(current_head, next_move):
    """
    return the coordinate of the head if our snake goes that way
    """
    MOVE_LOOKUP = {"left":-1, "right": 1, "up": 1, "down":-1}
    # print(f'Current head: {current_head}')
    # Copy first
    future_head = current_head.copy()
    if next_move in ["left", "right"]:
        # X-axis
        future_head["x"] = current_head["x"] + MOVE_LOOKUP[next_move]
    elif next_move in ["up", "down"]:
        future_head["y"] = current_head["y"] + MOVE_LOOKUP[next_move]

    return future_head

def avoid_walls(future_head, board_width, board_height):
    result = True

    x = int(future_head["x"])
    y = int(future_head["y"])

    if x < 0 or y < 0 or x >= board_width or y >= board_height:
        result = False

    return result

def avoid_snakes(future_head, snake_bodies):
    for snake in snake_bodies:
        if future_head in snake["body"][:-1]:
            return False
    return True

# adapted from https://github.com/altersaddle/untimely-neglected-wearable
def get_safe_moves(possible_moves, body, board):
    safe_moves = []
    for guess in possible_moves:
        guess_coord = get_next(body[0], guess)
        if avoid_walls(guess_coord, board["width"], board["height"]) and avoid_snakes(guess_coord, board["snakes"]): 
            safe_moves.append(guess)
        # elif len(body) > 1 and guess_coord not in body[:-1]:
        #    # The tail is also a safe place to go... unless there is a non-tail segment there too
        #    safe_moves.append(guess)
    # print(safe_moves)
    return safe_moves

def heurestic_safe_moves(game_state, id):
    possible_moves = ["up", "down", "left", "right"]
    safe_moves = get_safe_moves(possible_moves, get_current_id_snake_body(game_state, id), game_state['board'])
    return safe_moves

def heurestic_dist_to_food(game_state, id):
    food_distances = []
    food_locations = generate_np_food_array(game_state)
    head_location = generate_np_snake_head_array(game_state, id)
    for i in food_locations:
        food_distances.append(np.linalg.norm(i - head_location))
    return food_distances

def heuristic_check_if_ontop_food(game_state, id):
    head = get_current_id_snake_head(game_state, id)
    food_array = generate_np_food_array(game_state)
    for i in food_array:
        if np.count_nonzero([head == food_array]): 
            return True
    return False

def heuristic_check_if_ontop_snake(game_state, id):
    head = generate_np_snake_head_array(game_state, id)
    body = generate_np_snake_body_array(game_state, id)
    for i in body:
        if np.count_nonzero([head == i]):
            return True
    return False
     
def heurestic(game_state, maximizingPlayer):

    value = 0
    id = get_my_id(game_state)
    # combine various heurestic helper functions here
    safe_moves = heurestic_safe_moves(game_state, id)
    # occupied_board = generate_np_matrix(game_state)
    is_ontop_snake = heuristic_check_if_ontop_snake(game_state, id)
    dist_ontop_snake = 0
    if is_ontop_snake:
        dist_ontop_snake = 1
    else:
        is_ontop_snake = 0
    dist_to_food = heurestic_dist_to_food(game_state, id)
    dist_to_food = min(dist_to_food)
    if dist_to_food == 0:
        dist_to_food = 0.5
    value = 2 * len(safe_moves) + \
            2 * (1 / dist_to_food) + \
            -10 * dist_ontop_snake
    return value, maximizingPlayer

def update_player_body_minimax(game_state, id, head):
    body = get_current_id_snake_body(game_state=game_state, id=id)
    new_body = body.copy()
    new_body.insert(0, head)
    new_body.pop(-1)
    for i in game_state['board']['snakes']:
        if i['id'] == id:
            i['body'] = new_body
            i['head'] = head
            break
    
    return game_state

def generate_np_snake_body_array(game_state, id):
    body_list = []
    for i in game_state['board']['snakes']:
        if i['id'] == id:
            for j in i['body']:
                body_list.append(np.array([j['x'], j['y']]))
    return body_list

def generate_np_food_array(game_state):
    return [np.array([i['x'],i['y']]) for i in game_state['board']['food']]

def generate_np_snake_head_array(game_state, id):
    return [np.array([i['head']['x'],i['head']['y']]) for i in game_state['board']['snakes'] if i['id'] == id]

def generate_np_matrix(game_state):
    row, col = game_state['board']['width'], game_state['board']['height']
    food_list = [i for i in game_state['board']['food']]
    my_id = get_my_id(game_state)
    other_snakes = get_other_snakes(game_state)
    np_matrix = np.zeros((row, col))
    # place all the food on the map
    for i in food_list:
        np_matrix[i['x'], i['y']] = 1
    # place all the occupied snake tiles on the map
    for i in other_snakes:
        for j in i['body']:
            np_matrix[j['x'], j['y']] = 99
    for i in get_current_id_snake_body(game_state, my_id):
        np_matrix[i['x'], i['y']] = 99
    return np_matrix

def get_my_id(game_state):
    return game_state['you']['id'] 

def get_other_snakes(game_state):
    return [i for i in game_state['board']['snakes'] if i['id'] != get_my_id(game_state)]

def get_current_id_snake_body(game_state, id):
    return [i['body'] for i in game_state['board']['snakes'] if i['id'] == id][0]

def get_current_id_snake_head(game_state, id):
    return [i['head'] for i in game_state['board']['snakes'] if i['id'] == id][0]

def minimax(game_state, depth, maximizingPlayer):

    possible_moves = ["up", "down", "left", "right"]
    if depth == 0:
        return heurestic(game_state, maximizingPlayer)
    if maximizingPlayer:
        my_id = get_my_id(game_state)
        bestValue = -math.inf
        bestMove = None
        for j in get_safe_moves(possible_moves, get_current_id_snake_body(game_state, my_id), game_state['board']):
            next_head = get_next(get_current_id_snake_head(game_state=game_state, id=my_id), j)
            # check to see if we are on top food
            check_food = heuristic_check_if_ontop_food(game_state, my_id)
            if check_food:
                bestValue = 100
                bestMove = j
                return bestValue, bestMove
            if avoid_walls(next_head, game_state['board']['width'], game_state['board']['height']):
                game_state = update_player_body_minimax(game_state=game_state, id=my_id, head=next_head)
                new_value, _ = minimax(game_state=game_state, depth=depth-1,  maximizingPlayer=False)
                if max(new_value, bestValue) > bestValue:
                    bestValue = new_value
                    bestMove = j
        return bestValue, bestMove
    else:
        other_ids = get_other_snakes(game_state=game_state)
        bestValue = math.inf
        bestMove = None
        for i in other_ids:
            id = i['id']
            for j in get_safe_moves(possible_moves, get_current_id_snake_body(game_state, id), game_state['board']):
                next_head = get_next(get_current_id_snake_head(game_state=game_state, id=id), j)
                if avoid_walls(next_head, game_state['board']['width'], game_state['board']['height']):
                    game_state = update_player_body_minimax(game_state=game_state, id=id, head=next_head)
                    new_value, _ = minimax(game_state=game_state, depth=depth-1, maximizingPlayer=True)
                    if min(new_value, bestValue) < bestValue:
                        bestValue = new_value
                        bestMove = j
        return bestValue, bestMove

# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    my_id = get_my_id(game_state=game_state)
    
    
    value, next_move = minimax(game_state=game_state, depth=5, maximizingPlayer=True)

    print(f"MOVE {game_state['turn']}: {next_move}. Value: {value}")
    return {"move": next_move}

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    port = "8000"
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == '--port':
            port = sys.argv[i+1]
        elif sys.argv[i] == '--seed':
            random_seed = int(sys.argv[i+1])
    run_server({"info": info, "start": start, "move": move, "end": end, "port": port})
