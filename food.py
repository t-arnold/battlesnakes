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
import sys


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")

def move_towards_food(game_state: typing.Dict, safe_moves: list, my_head: typing.Dict) -> str:
    """
    Move towards the closest food if possible, considering only safe moves.
    """
    food_positions = game_state['board']['food']
    closest_food = None
    min_distance = float('inf')

    # Find the closest piece of food
    for food in food_positions:
        distance = abs(food['x'] - my_head['x']) + abs(food['y'] - my_head['y'])
        if distance < min_distance:
            min_distance = distance
            closest_food = food

    if closest_food:
        # Determine direction towards closest food
        if closest_food['x'] < my_head['x'] and "left" in safe_moves:
            return "left"
        elif closest_food['x'] > my_head['x'] and "right" in safe_moves:
            return "right"
        elif closest_food['y'] < my_head['y'] and "down" in safe_moves:
            return "down"
        elif closest_food['y'] > my_head['y'] and "up" in safe_moves:
            return "up"

    # If moving directly towards food is not safe, return None
    return None


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # TODO: Step 1 - Prevent your Battlesnake from moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    # Check for out-of-bounds moves
    if my_head["x"] == 0: is_move_safe["left"] = False
    if my_head["x"] == board_width - 1: is_move_safe["right"] = False
    if my_head["y"] == 0: is_move_safe["down"] = False
    if my_head["y"] == board_height - 1: is_move_safe["up"] = False

    # TODO: Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state['you']['body']
    for move, isSafe in is_move_safe.items():
        if not isSafe: continue  # Skip already unsafe moves
        future_position = get_future_position(my_head, move)
        if future_position in my_body:
            is_move_safe[move] = False

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']
    for move, isSafe in is_move_safe.items():
        if not isSafe: continue  # Skip already unsafe moves
        future_position = get_future_position(my_head, move)
        for snake in opponents:
            if future_position in snake['body']:
                is_move_safe[move] = False

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Choose a random move from the safe ones
    next_move = random.choice(safe_moves)

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    # food = game_state['board']['food']

    # print(f"MOVE {game_state['turn']}: {next_move}")
    # return {"move": next_move}
    if len(safe_moves) > 0:
        food_move = move_towards_food(game_state, safe_moves, my_head)
        if food_move:
            next_move = food_move
        else:
            # If there is no safe move towards food, choose a random safe move
            next_move = random.choice(safe_moves)
    else:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        next_move = "down"

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

def get_future_position(current_position, direction):
    future_position = current_position.copy()
    if direction == "up": future_position["y"] += 1
    elif direction == "down": future_position["y"] -= 1
    elif direction == "left": future_position["x"] -= 1
    elif direction == "right": future_position["x"] += 1
    return future_position

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
    port = "8000"
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == '--port':
            port = sys.argv[i+1]

    run_server({"info": info, "start": start, "move": move, "end": end, "port": port})
