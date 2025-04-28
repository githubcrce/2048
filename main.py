import random
import os

# --- Game Setup ---

def init_board():
    board = [[0 for _ in range(4)] for _ in range(4)]
    add_random_tile(board)
    add_random_tile(board)
    return board

def add_random_tile(board):
    empty = [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]
    if not empty:
        return
    i, j = random.choice(empty)
    board[i][j] = 2 if random.random() < 0.9 else 4

# --- Board Display ---

def print_board(board):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("2048 Game\n")
    for row in board:
        print("+------+------+------+------+" )
        print("|" + "|".join(f"{num:^6}" if num != 0 else "      " for num in row) + "|")
    print("+------+------+------+------+")

# --- Move Mechanics ---

def merge(row):
    new_row = [i for i in row if i != 0]
    for i in range(len(new_row)-1):
        if new_row[i] == new_row[i+1]:
            new_row[i] *= 2
            new_row[i+1] = 0
    new_row = [i for i in new_row if i != 0]
    return new_row + [0]*(4 - len(new_row))

def move_left(board):
    return [merge(row) for row in board]

def move_right(board):
    return [list(reversed(merge(reversed(row)))) for row in board]

def transpose(board):
    return [list(row) for row in zip(*board)]

def move_up(board):
    board = transpose(board)
    board = move_left(board)
    return transpose(board)

def move_down(board):
    board = transpose(board)
    board = move_right(board)
    return transpose(board)

# --- Game Status ---

def is_game_over(board):
    for move in [move_left, move_right, move_up, move_down]:
        if board != move([row[:] for row in board]):
            return False
    return True

# --- Main Loop ---

def main():
    board = init_board()
    while True:
        print_board(board)
        move = input("Enter move (W/A/S/D): ").lower()
        if move not in ('w', 'a', 's', 'd'):
            continue

        moves = {'w': move_up, 'a': move_left, 's': move_down, 'd': move_right}
        new_board = moves[move]([row[:] for row in board])

        if new_board != board:
            board = new_board
            add_random_tile(board)

        if is_game_over(board):
            print_board(board)
            print("Game Over!")
            break

if __name__ == "__main__":
    main()
