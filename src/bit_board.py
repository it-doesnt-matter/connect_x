import copy
from collections.abc import Iterator
from time import perf_counter

from kaggle_environments import make
from kaggle_environments.utils import Struct


class BitBoard:
    def __init__(self, width: int = 7, height: int = 6) -> None:
        self.width = width
        self.height = height
        self.own_marks = 0
        self.all_marks = 0

    @classmethod
    def from_move_sequence(cls, moves: list[int]) -> "BitBoard":
        board = BitBoard()
        for move in moves:
            board.make_move(move)
        return board

    @classmethod
    def from_observation(cls, observation: Struct, configuration: Struct) -> "BitBoard":
        width = configuration.columns
        height = configuration.rows
        own_marks = [0] * width * (height + 1)
        all_marks = [0] * width * (height + 1)

        player = observation.mark
        opponent = 1 if player == 2 else 2
        for i, cell in enumerate(observation.board):
            if cell == player:
                j = (height - 1) - (i // width) + ((i % width) * (height + 1))
                own_marks[j] = 1
                all_marks[j] = 1
            elif cell == opponent:
                j = (height - 1) - (i // width) + ((i % width) * (height + 1))
                all_marks[j] = 1

        board = BitBoard(width, height)
        own_marks = "".join(map(str, own_marks))
        all_marks = "".join(map(str, all_marks))
        board.own_marks = int(own_marks[::-1], 2)
        board.all_marks = int(all_marks[::-1], 2)
        return board

    def is_possible_move(self, column: int) -> bool:
        return (self.all_marks & self.get_top_of_column_mask(column)) == 0

    # this doesn't check if the move is possible
    def make_move(self, column: int) -> None:
        # this is a bit board with only one bit representing the move to make
        move = self.all_marks + self.get_bottom_of_column_mask(column)
        move &= self.get_column_mask(column)
        self.own_marks ^= self.all_marks
        self.all_marks |= move

    def get_column_mask(self, column: int) -> int:
        return ((1 << self.height) - 1) << (column * (self.height + 1))

    def get_bottom_of_column_mask(self, column: int) -> int:
        return 1 << (column * (self.height + 1))

    def get_bottom_row_mask(self) -> int:
        start = 0
        for i in range(1, self.width + 1):
            start |= 1 << ((self.width - i) * (self.height + 1))
        return start

    # top refers to the top of the game board not the bit board
    def get_top_of_column_mask(self, column: int) -> int:
        return 1 << ((self.height - 1) + column * (self.height + 1))

    def get_board_mask(self) -> int:
        return self.get_bottom_row_mask() * ((1 << self.height) - 1)

    def is_winning_state(self, column: int) -> bool:
        return (
            self.get_winning_positions()
            & self.get_possible_moves()
            & self.get_column_mask(column)
        )

    def enables_opponent_win(self, column: int) -> bool:
        return (
            self.get_losing_positions() >> 1
            & self.get_possible_moves()
            & self.get_column_mask(column)
        )

    # this returns a bit board with all possible moves
    # losing moves are also included
    def get_possible_moves(self) -> int:
        return (self.all_marks + self.get_bottom_row_mask()) & self.get_board_mask()

    def get_possible_three_streaks(self, position: int, mask: int) -> int:
        # vertical checks
        r = (position << 1) & (position << 2)
        # horizontal checks
        p = position << (self.height + 1)
        r |= p & (position << (2 * (self.height + 1)))
        r |= p & (position >> (self.height + 1))
        p = position >> (self.height + 1)
        r |= p & (position << (self.height + 1))
        r |= p & (position >> (2 * (self.height + 1)))
        # diagonal checks
        p = position << self.height
        r |= p & (position << (2 * self.height))
        r |= p & (position >> self.height)
        p = position >> self.height
        r |= p & (position << self.height)
        r |= p & (position >> (2 * self.height))
        # anti-diagonal checks
        p = position << (self.height + 2)
        r |= p & (position << (2 * (self.height + 2)))
        r |= p & (position >> (self.height + 2))
        p = position >> (self.height + 2)
        r |= p & (position << (self.height + 2))
        r |= p & (position >> (2 * (self.height + 2)))

        return r & (self.get_board_mask() ^ mask)

    def get_possible_four_streaks(self, position: int, mask: int) -> int:
        # vertical checks
        r = (position << 1) & (position << 2) & (position << 3)
        # horizontal checks
        p = (position << (self.height + 1)) & (position << (2 * (self.height + 1)))
        r |= p & (position << (3 * (self.height + 1)))
        r |= p & (position >> (self.height + 1))
        p = (position >> (self.height + 1)) & (position >> (2 * (self.height + 1)))
        r |= p & (position << (self.height + 1))
        r |= p & (position >> (3 * (self.height + 1)))
        # diagonal checks
        p = (position << self.height) & (position << (2 * self.height))
        r |= p & (position << (3 * self.height))
        r |= p & (position >> self.height)
        p = (position >> self.height) & (position >> (2 * self.height))
        r |= p & (position << self.height)
        r |= p & (position >> (3 * self.height))
        # anti-diagonal checks
        p = (position << (self.height + 2)) & (position << (2 * (self.height + 2)))
        r |= p & (position << (3 * (self.height + 2)))
        r |= p & (position >> (self.height + 2))
        p = (position >> (self.height + 2)) & (position >> (2 * (self.height + 2)))
        r |= p & (position << (self.height + 2))
        r |= p & (position >> (3 * (self.height + 2)))

        return r & (self.get_board_mask() ^ mask)

    # not configuration agnostic
    def get_winning_positions(self) -> int:
        return self.get_possible_four_streaks(self.own_marks, self.all_marks)

    # not configuration agnostic
    def get_losing_positions(self) -> int:
        return self.get_possible_four_streaks(self.own_marks ^ self.all_marks, self.all_marks)

    # not configuration agnostic
    def get_move_score(self, move: int) -> int:
        winning_positions = self.get_possible_four_streaks(self.own_marks | move, self.all_marks)
        return self.popcount(winning_positions)

    # not configuration agnostic
    # this returns the score of the board and is used as heuristic in negamax
    def get_board_score(self) -> int:
        own_score = self._get_player_score(self.own_marks, self.all_marks)
        opponent_score = self._get_player_score(self.own_marks ^ self.all_marks, self.all_marks)
        return own_score - opponent_score

    # not configuration agnostic
    # this returns the score of only one of the two players
    def _get_player_score(self, position: int, mask: int) -> int:
        three_in_a_row = self.get_possible_three_streaks(position, mask)
        four_in_a_row = self.get_possible_four_streaks(position, mask)
        four_mask = four_in_a_row ^ self.get_board_mask()
        exclusive_three_in_a_row = four_mask & three_in_a_row

        three_count = self.popcount(exclusive_three_in_a_row)
        four_count = self.popcount(four_in_a_row)

        return four_count * 10 + three_count * 1

    def visualize_board(self) -> None:
        opponent = self.own_marks ^ self.all_marks
        opponent = f"{opponent:0>49b}"
        opponent = opponent[::-1]
        player = f"{self.own_marks:0>49b}"
        player = player[::-1]

        visualization = ["\u2554"]
        visualization.append("\u2550" * (2 * self.width + 1))
        visualization.append("\u2557\n")
        for i in range(self.height, -1, -1):
            visualization.append("\u2551 ")
            for j in range(0, self.width * (self.height + 1), self.height + 1):
                cell = i + j
                char = "."
                if player[cell] == "1":
                    char = "X"
                elif opponent[cell] == "1":
                    char = "0"
                visualization.append(f"{char} ")
            visualization.append("\u2551\n")
        visualization.append("\u255A")
        visualization.append("\u2550" * (2 * self.width + 1))
        visualization.append("\u255D")
        visualization = "".join(visualization)
        print(visualization)

    def popcount(self, m: int) -> int:
        count = 0
        while count < m:
            m &= m - 1
            count += 1
        return count


class OrderedMoveList:
    def __init__(self, board: BitBoard) -> None:
        self.moves = []
        # this is used for ties, in which case the column closer to the middle wins
        possible_moves = board.get_possible_moves()
        column_order = []

        for i in range(board.width):
            column_order.append((board.width // 2) + ((1 - 2 * (i % 2)) * (i + 1) // 2))

        for i in reversed(range(board.width)):
            column = column_order[i]
            move = possible_moves & board.get_column_mask(column)
            if move == 0 or board.enables_opponent_win(column):
                continue
            score = board.get_move_score(move)
            self.append(column, score)

        # this ensure that the list is not empty in case that there
        # are no possible non-losing moves
        if not self.moves:
            for column in range(board.width):
                if board.is_possible_move(column):
                    self.append(column, 0)
                    break

    def __iter__(self) -> Iterator[int]:
        for move in reversed(self.moves):
            yield move[0]

    def append(self, column: int, score: int) -> None:
        size = len(self.moves)
        self.moves.append(None)
        position = 0
        for position in reversed(range(size)):
            if self.moves[position][1] > score:
                self.moves[position + 1] = self.moves[position]
            else:
                position += 1
                break
        self.moves[position] = (column, score)

    def pop(self) -> int:
        if self.moves:
            return self.moves.pop()[0]
        else:
            return -1


# not configuration agnostic
def negamax(board: BitBoard, depth: int, alpha: int, beta: int) -> tuple[int, int]:
    if depth == 0:
        return (board.get_board_score(), 0)

    for column in range(board.width):
        if board.is_winning_state(column):
            return (10 ** 4, column)

    best_score = - 10 ** 4
    best_column = 3
    for column in range(board.width):
        if board.is_possible_move(column):
            best_column = column
            break
    moves = OrderedMoveList(board)

    for column in moves:
        child = copy.deepcopy(board)
        child.make_move(column)

        child_score, _ = negamax(child, depth - 1, -beta, -alpha)
        child_score *= -1

        if child_score > best_score:
            best_score = child_score
            best_column = column

        alpha = max(alpha, best_score)
        if alpha >= beta:
            break

    return (best_score, best_column)


def negamax_agent(observation: Struct, configuration: Struct) -> int:
    start_time = perf_counter()
    board = BitBoard.from_observation(observation, configuration)
    _, column = negamax(board, 10, -1_000_000, 1_000_000)
    end_time = perf_counter()
    print(f"{end_time - start_time:0.4f}")
    return column


def visualize_single_side(bit_board: int) -> None:
    bit_board = f"{bit_board:0>49b}"
    bit_board = bit_board[::-1]
    visualization = ["\u2554"]
    visualization.append("\u2550" * (2 * 7 + 1))
    visualization.append("\u2557\n")
    for i in range(6, -1, -1):
        visualization.append("\u2551 ")
        for j in range(0, 7 * (6 + 1), 6 + 1):
            cell = i + j
            char = "X" if bit_board[cell] == "1" else "O" if bit_board[cell] == "2" else "."
            visualization.append(f"{char} ")
        visualization.append("\u2551\n")
    visualization.append("\u255A")
    visualization.append("\u2550" * (2 * 7 + 1))
    visualization.append("\u255D")
    visualization = "".join(visualization)
    print(visualization)


def main() -> None:
    env = make("connectx", {"rows": 6, "columns": 7, "inarow": 4}, debug=True)
    env.run([negamax_agent, "negamax"])
    html_output = env.render(mode="html")
    with open("last_game.html", "w") as file:
        file.write(html_output)


if __name__ == "__main__":
    main()
