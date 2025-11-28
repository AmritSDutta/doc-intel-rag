import asyncio
import re

from llama_index.llms.google_genai import GoogleGenAI
from workflows import Workflow, step
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
)

from llama_index.llms.openai import OpenAI
from workflows.resource import ResourceManager

PLAYER_TEMPLATE = """
You are an autonomous Tic-Tac-Toe agent.

YOUR SYMBOL: {{SYMBOL}}
BOARD STATE:
{{BOARD}}

GENERAL RULES:
1. You play exactly one move per turn.
2. A move must target a cell that currently contains '.' (empty).
3. Choose the strongest legal move for {{SYMBOL}} only.
4. Never choose a filled cell.
5. Never describe reasoning, analysis, or commentary.


OBJECTIVE:
Maximize your chance of winning and minimize opponent advantage.

WIN-MAXIMIZATION STRATEGY (APPLY IN ORDER):
1. Immediate Win: play any move that wins instantly.
2. Block Opponent: if opponent can win next turn, block that move.
3. Center: take (1,1) if empty.
4. Corners: take any available corner.
5. Best Available: choose the most advantageous remaining empty cell.

RULES:
- You must pick exactly one empty cell.
- Never select a filled square.
- No explanations.

OUTPUT FORMAT (STRICT):
Return only:

    row,col

No other text, punctuation, or formatting.
"""


def _parse_coord(s: str) -> tuple[int, int] | None:
    m = re.search(r"(-?\d+)\s*[, ]\s*(-?\d+)", s)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _valid_move(board, i, j):
    return 0 <= i < 3 and 0 <= j < 3 and board[i][j] == '.'


def _check_winner(b):
    lines = ([(r, c) for c in range(3)] for r in range(3))  # rows generator
    wins = []
    for r in range(3):
        if b[r][0] == b[r][1] == b[r][2] != '.':
            return b[r][0]
    for c in range(3):
        if b[0][c] == b[1][c] == b[2][c] != '.':
            return b[0][c]
    if b[0][0] == b[1][1] == b[2][2] != '.':
        return b[0][0]
    if b[0][2] == b[1][1] == b[2][0] != '.':
        return b[0][2]
    if all(cell != '.' for row in b for cell in row):
        return 'DRAW'
    return None


class CoordinatorEvent(Event):
    moves: tuple[int, int]
    player: str


class PlayerOneEvent(Event):
    player: str


class PlayerTwoEvent(Event):
    player: str


class TicTacToeFlow(Workflow):

    def __init__(self, timeout: float | None = 45.0, disable_validation: bool = False, verbose: bool = False,
                 resource_manager: ResourceManager | None = None, num_concurrent_runs: int | None = None) -> None:
        super().__init__(timeout, disable_validation, verbose, resource_manager, num_concurrent_runs)

        self.player1 = OpenAI(model="gpt-5-nano")
        self.player2 = GoogleGenAI(model="gemini-2.5-flash")
        self.board: list[list[str]] = [['.' for _ in range(3)] for _ in range(3)]

    @step
    async def player_1(self, ev: PlayerOneEvent) -> CoordinatorEvent:
        print('player one move:')
        prompt = (PLAYER_TEMPLATE
                  .replace("{{SYMBOL}}", "O")
                  .replace("{{BOARD}}", str(self.board)))

        response = await self.player1.acomplete(prompt)
        coord = _parse_coord(response.text)
        if coord is None:
            raise ValueError(f"unparsable move: {response}")
        return CoordinatorEvent(moves=coord, player='O')

    @step
    async def player_2(self, ev: PlayerTwoEvent) -> CoordinatorEvent:
        print('player two move:')

        prompt = (PLAYER_TEMPLATE
                  .replace("{{SYMBOL}}", "X")
                  .replace("{{BOARD}}", str(self.board)))
        response = await self.player2.acomplete(prompt)
        coord = _parse_coord(response.text)
        if coord is None:
            raise ValueError(f"unparsable move: {response.text}")
        return CoordinatorEvent(moves=coord, player='X')

    @step
    async def coordinator_decision(self,
                                   ev: StartEvent | CoordinatorEvent) -> StopEvent | PlayerOneEvent | PlayerTwoEvent:
        if isinstance(ev, StartEvent):
            return PlayerOneEvent(player='player1')

        moves, symbol = ev.moves, ev.player
        i, j = moves
        if not _valid_move(self.board, i, j):
            return StopEvent(result=f"Invalid move {(i, j)} by {ev.player}")
        self.board[i][j] = ev.player
        self._print_box()

        win = _check_winner(self.board)
        if win == 'DRAW':
            return StopEvent(result=f"DRAW. Board: {self.board}")
        if win in ('X', 'O'):
            return StopEvent(result=f"WINNER {win}. Final: {self.board}")

        next_event = PlayerOneEvent(player='player1') if ev.player == 'X' else PlayerTwoEvent(player='player2')
        return next_event

    def _print_box(self):
        for row in self.board:
            line = " ".join("   " if c == "." else f" {c} " for c in row)
            print(f"|{line}|")
        print('\n')

    def _valid(self, i, j):
        return 0 <= i < 3 and 0 <= j < 3 and self.board[i][j] == '.'


if __name__ == '__main__':
    async def main():
        grid = [['.' for _ in range(3)] for _ in range(3)]
        w = TicTacToeFlow(timeout=360, verbose=False)
        result = await w.run(board=grid)
        print(result)


    asyncio.run(main())
