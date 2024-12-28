"""
3D Tic-Tac-Toe with CUDA acceleration
Supports arbitrary NxNxN board sizes with efficient bit-based representation
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Set, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
from numba import cuda, uint64, int32, float32 # type: ignore
import math
import time
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from collections import deque
from statistics import mean

# Constants for CUDA optimization
WARP_SIZE = 32
MAX_THREADS_PER_BLOCK = 1024
MIN_BLOCKS_PER_SM = 2

class GameResult(Enum):
    """Possible game states"""
    WIN = 1
    LOSS = -1
    DRAW = 0
    IN_PROGRESS = 2

@dataclass(frozen=True)
class GameConfig:
    """Configuration for 3D Tic-Tac-Toe game"""
    size: int  # N for NxNxN board
    target: int  # Number in a row needed to win
    num_u64s: int = field(init=False)
    
    def __post_init__(self):
        assert isinstance(self.size, int), f"Size must be int, got {type(self.size)}"
        assert isinstance(self.target, int), f"Target must be int, got {type(self.target)}"
        assert self.size >= 3, f"Size must be at least 3, got {self.size}"
        assert 3 <= self.target <= self.size, f"Target must be between 3 and size, got {self.target}"
        
        # Calculate required uint64s
        total_positions = self.size ** 3
        # Use object.__setattr__ to set frozen dataclass field
        object.__setattr__(self, 'num_u64s', (total_positions + 63) // 64)

class BitBoard:
    """Efficient board representation using multiple uint64s"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.size
        self.num_u64s = config.num_u64s
        self.bits = np.zeros(self.num_u64s, dtype=np.uint64)
    
    def _get_indices(self, x: int, y: int, z: int) -> Tuple[int, int]:
        """Convert 3D coordinates to array index and bit position"""
        pos = x + y * self.size + z * self.size * self.size
        return pos // 64, pos % 64
    
    def set_bit(self, x: int, y: int, z: int) -> None:
        """Set a piece at the given coordinates"""
        array_idx, bit_idx = self._get_indices(x, y, z)
        self.bits[array_idx] |= np.uint64(1) << bit_idx
    
    def get_bit(self, x: int, y: int, z: int) -> bool:
        """Check if a piece exists at the given coordinates"""
        array_idx, bit_idx = self._get_indices(x, y, z)
        return bool(self.bits[array_idx] & (np.uint64(1) << bit_idx))
    
    def clear_bit(self, x: int, y: int, z: int) -> None:
        """Clear a piece at the given coordinates"""
        array_idx, bit_idx = self._get_indices(x, y, z)
        self.bits[array_idx] &= ~(np.uint64(1) << bit_idx)
    
    def is_empty(self) -> bool:
        """Check if board is empty"""
        return not np.any(self.bits)
    
    def count_bits(self) -> int:
        """Count total number of pieces"""
        return sum(bin(x).count('1') for x in self.bits)
    
    def __or__(self, other: 'BitBoard') -> 'BitBoard':
        """Bitwise OR of two boards"""
        result = BitBoard(self.config)
        result.bits = self.bits | other.bits
        return result
    
    def __and__(self, other: 'BitBoard') -> 'BitBoard':
        """Bitwise AND of two boards"""
        result = BitBoard(self.config)
        result.bits = self.bits & other.bits
        return result
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitBoard):
            return NotImplemented
        return np.array_equal(self.bits, other.bits)

class WinPatternGenerator:
    """Generates winning patterns for arbitrary board sizes"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.size
        self.target = config.target
        
        # Pre-compute directions for efficiency
        self.directions = [
            # Straight lines
            (1,0,0), (0,1,0), (0,0,1),
            # 2D diagonals
            (1,1,0), (1,-1,0), (1,0,1),
            (1,0,-1), (0,1,1), (0,1,-1),
            # 3D diagonals
            (1,1,1), (1,1,-1), (1,-1,1), (-1,1,1),
            (1,-1,-1), (-1,1,-1), (-1,-1,1)
        ]
    
    def generate_pattern(self, start_x: int, start_y: int, start_z: int,
                        dx: int, dy: int, dz: int) -> Optional[BitBoard]:
        """Generate a single winning pattern"""
        pattern = BitBoard(self.config)
        count = 0
        
        for i in range(self.target):
            x, y, z = start_x + dx*i, start_y + dy*i, start_z + dz*i
            if not (0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size):
                return None
            pattern.set_bit(x, y, z)
            count += 1
        
        return pattern if count == self.target else None
    
    def generate_all_patterns(self) -> List[BitBoard]:
        """Generate all possible winning patterns"""
        patterns: List[BitBoard] = []
        
        # Generate patterns for all starting positions and directions
        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    for dx, dy, dz in self.directions:
                        pattern = self.generate_pattern(x, y, z, dx, dy, dz)
                        if pattern is not None:
                            patterns.append(pattern)
        
        assert patterns, "No valid patterns generated"
        return patterns

@cuda.jit(device=True)
def check_pattern_match(board_bits: uint64[:], pattern_bits: uint64[:],  # type: ignore
                       n_u64s: int32) -> bool: # type: ignore
    """Device function to check if a pattern matches a board position"""
    for i in range(n_u64s):
        if pattern_bits[i] and (board_bits[i] & pattern_bits[i]) != pattern_bits[i]:
            return False
    return True

@cuda.jit
def evaluate_boards_kernel(boards_p1, boards_p2, results, patterns, 
                         n_patterns: int32, n_u64s: int32): # type: ignore
    """CUDA kernel for board evaluation"""
    idx = cuda.grid(1)
    if idx >= boards_p1.shape[0]:
        return
    
    # Check each pattern
    for pattern_idx in range(n_patterns):
        pattern_offset = pattern_idx * n_u64s
        
        # Check player 1 win
        if check_pattern_match(
            boards_p1[idx], 
            patterns[pattern_offset:pattern_offset + n_u64s],
            n_u64s
        ):
            results[idx] = GameResult.WIN.value
            return
            
        # Check player 2 win
        if check_pattern_match(
            boards_p2[idx],
            patterns[pattern_offset:pattern_offset + n_u64s],
            n_u64s
        ):
            results[idx] = GameResult.LOSS.value
            return
    
    # Check for draw
    is_full = True
    for u64_idx in range(n_u64s):
        if (boards_p1[idx, u64_idx] | boards_p2[idx, u64_idx]) != ~np.uint64(0):
            is_full = False
            break
    
    results[idx] = GameResult.DRAW.value if is_full else GameResult.IN_PROGRESS.value

class TicTacToe3D:
    """Main game class for arbitrary-sized 3D Tic-Tac-Toe"""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.size
        self.num_u64s = config.num_u64s
        
        # Generate winning patterns
        generator = WinPatternGenerator(config)
        self.patterns = generator.generate_all_patterns()
        
        # Convert patterns to CUDA-friendly format
        pattern_array = np.zeros((len(self.patterns) * self.num_u64s), dtype=np.uint64)
        for i, pattern in enumerate(self.patterns):
            pattern_array[i * self.num_u64s:(i + 1) * self.num_u64s] = pattern.bits
        
        # CUDA setup and optimization
        device = cuda.get_current_device()
        self.max_threads_per_block = min(device.MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK)
        self.warp_size = WARP_SIZE
        
        # Calculate optimal thread count (multiple of warp size)
        self.threadsperblock = min(
            self.max_threads_per_block,
            ((self.max_threads_per_block // self.warp_size) * self.warp_size)
        )
        
        # Create pinned memory for patterns
        self.pattern_array = cuda.pinned_array(pattern_array.shape, dtype=np.uint64)
        self.pattern_array[:] = pattern_array
        self.d_patterns = cuda.to_device(self.pattern_array)
        
        # Initialize empty boards for new game
        self.board_p1 = BitBoard(config)
        self.board_p2 = BitBoard(config)
        self.current_player = 1
        
        # Preallocate pinned memory for batch evaluation
        self.max_batch_size = 1024  # Adjust based on your needs
        self.h_boards_p1 = cuda.pinned_array((self.max_batch_size, self.num_u64s), dtype=np.uint64)
        self.h_boards_p2 = cuda.pinned_array((self.max_batch_size, self.num_u64s), dtype=np.uint64)
    
    def evaluate_batch(self, boards_p1: List[BitBoard], 
                      boards_p2: List[BitBoard]) -> npt.NDArray[np.int32]:
        """Evaluate multiple board positions in parallel"""
        n_boards = len(boards_p1)
        assert len(boards_p2) == n_boards, "Must provide equal number of boards"
        assert n_boards <= self.max_batch_size, f"Batch size {n_boards} exceeds maximum {self.max_batch_size}"
        
        # Copy boards to pinned memory
        for i in range(n_boards):
            self.h_boards_p1[i] = boards_p1[i].bits
            self.h_boards_p2[i] = boards_p2[i].bits
        
        # Transfer to device using pinned memory
        d_boards_p1 = cuda.to_device(self.h_boards_p1[:n_boards])
        d_boards_p2 = cuda.to_device(self.h_boards_p2[:n_boards])
        d_results = cuda.device_array(n_boards, dtype=np.int32)
        
        # Calculate optimal grid size
        min_grid_size = (n_boards + self.threadsperblock - 1) // self.threadsperblock
        device = cuda.get_current_device()
        max_blocks = device.MULTIPROCESSOR_COUNT * MIN_BLOCKS_PER_SM
        blockspergrid = max(min_grid_size, max_blocks)
        
        # Launch kernel
        evaluate_boards_kernel[blockspergrid, self.threadsperblock](
            d_boards_p1, d_boards_p2, d_results,
            self.d_patterns, np.int32(len(self.patterns)), np.int32(self.num_u64s)
        )
        
        return d_results.copy_to_host()

    def get_valid_moves(self, board_p1: BitBoard, board_p2: BitBoard) -> List[Tuple[int, int, int]]:
        """Get list of valid moves"""
        moves = []
        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    if not (board_p1.get_bit(x, y, z) or board_p2.get_bit(x, y, z)):
                        moves.append((x, y, z))
        return moves
    
    def make_move(self, x: int, y: int, z: int) -> bool:
        """Make a move at the specified position"""
        if not (0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size):
            raise ValueError(f"Invalid coordinates: ({x}, {y}, {z})")
        
        # Check if position is empty
        if self.board_p1.get_bit(x, y, z) or self.board_p2.get_bit(x, y, z):
            return False
        
        # Make the move
        if self.current_player == 1:
            self.board_p1.set_bit(x, y, z)
        else:
            self.board_p2.set_bit(x, y, z)
        
        self.current_player = 3 - self.current_player  # Switch players (1->2 or 2->1)
        return True
    
    def get_game_state(self) -> GameResult:
        """Get current game state"""
        result = self.evaluate_batch([self.board_p1], [self.board_p2])[0]
        return GameResult(result)
    
    def minimax(self, board_p1: BitBoard, board_p2: BitBoard, depth: int,
                alpha: float = float('-inf'), beta: float = float('inf'),
                maximizing: bool = True) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """Minimax algorithm with alpha-beta pruning"""
        result = self.evaluate_batch([board_p1], [board_p2])[0]
        
        if result == GameResult.WIN.value:
            return 1.0, None
        elif result == GameResult.LOSS.value:
            return -1.0, None
        elif result == GameResult.DRAW.value:
            return 0.0, None
        elif depth == 0:
            return self.heuristic(board_p1, board_p2), None
        
        moves = self.get_valid_moves(board_p1, board_p2)
        if not moves:
            return 0.0, None
        
        # Create reusable BitBoard objects for move simulation
        temp_board = BitBoard(self.config)
        
        # Score moves for better ordering
        move_scores = []
        for move in moves:
            temp_board.bits = board_p1.bits.copy() if maximizing else board_p2.bits.copy()
            temp_board.set_bit(*move)
            score = self.heuristic(temp_board if maximizing else board_p1,
                                 board_p2 if maximizing else temp_board)
            move_scores.append((score, move))
        
        # Sort moves by score (descending for maximizing, ascending for minimizing)
        move_scores.sort(reverse=maximizing)
        moves = [move for _, move in move_scores]
        
        best_move = None
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                # Reuse BitBoard object
                temp_board.bits = board_p1.bits.copy()
                temp_board.set_bit(*move)
                
                eval, _ = self.minimax(temp_board, board_p2, depth-1, alpha, beta, False)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                # Reuse BitBoard object
                temp_board.bits = board_p2.bits.copy()
                temp_board.set_bit(*move)
                
                eval, _ = self.minimax(board_p1, temp_board, depth-1, alpha, beta, True)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
    
    def heuristic(self, board_p1: BitBoard, board_p2: BitBoard) -> float:
        """Heuristic evaluation function"""
        # Count number of potential winning lines
        score = 0.0
        for pattern in self.patterns:
            p1_match = True
            p2_match = True
            p1_count = 0
            p2_count = 0
            
            for i in range(self.num_u64s):
                if pattern.bits[i]:
                    if board_p2.bits[i] & pattern.bits[i]:
                        p1_match = False
                    elif board_p1.bits[i] & pattern.bits[i]:
                        p1_count += bin(board_p1.bits[i] & pattern.bits[i]).count('1')
                    
                    if board_p1.bits[i] & pattern.bits[i]:
                        p2_match = False
                    elif board_p2.bits[i] & pattern.bits[i]:
                        p2_count += bin(board_p2.bits[i] & pattern.bits[i]).count('1')
            
            if p1_match:
                score += p1_count / self.config.target
            if p2_match:
                score -= p2_count / self.config.target
        
        return score / len(self.patterns)  # Normalize by number of patterns
    
    def get_best_move(self, depth: int = 4) -> Optional[Tuple[int, int, int]]:
        """Get best move using minimax with parallel evaluation"""
        moves = self.get_valid_moves(self.board_p1, self.board_p2)
        if not moves:
            return None
            
        # Always use minimax for move selection
        _, move = self.minimax(self.board_p1, self.board_p2, depth)
        return move
    
    def print_board(self) -> None:
        """Print current board state"""
        for z in range(self.size):
            print(f"\nLevel {z}")
            for y in range(self.size):
                for x in range(self.size):
                    if self.board_p1.get_bit(x, y, z):
                        print("X", end=" ")
                    elif self.board_p2.get_bit(x, y, z):
                        print("O", end=" ")
                    else:
                        print(".", end=" ")
                print()
            print()

def create_board_table(game: TicTacToe3D) -> Table:
    """Create a rich table representation of the board"""
    table = Table(show_header=False, show_edge=False, padding=0, box=None)
    
    # Add columns for each level
    for _ in range(game.size):
        table.add_column(justify="center", width=game.size * 2 + 1)
    
    # Add rows for each level
    for y in range(game.size):
        row = []
        for z in range(game.size):
            level = ""
            for x in range(game.size):
                if game.board_p1.get_bit(x, y, z):
                    level += "X "
                elif game.board_p2.get_bit(x, y, z):
                    level += "O "
                else:
                    level += ". "
            row.append(level.strip())
        table.add_row(*row)
    
    return table

def create_stats_panel(moves_per_sec: float, evals_per_sec: float, 
                      total_moves: int, total_evals: int,
                      move_times: List[float], eval_times: List[float]) -> Panel:
    """Create a panel with performance statistics"""
    stats = Text()
    stats.append("Performance Metrics\n\n", style="bold")
    stats.append(f"Moves/sec: {moves_per_sec:.2f}\n")
    stats.append(f"Evals/sec: {evals_per_sec:.2f}\n")
    stats.append(f"Total moves: {total_moves}\n")
    stats.append(f"Total evals: {total_evals}\n")
    
    # Handle empty data cases
    avg_move_time = mean(move_times)*1000 if move_times else 0.0
    avg_eval_time = mean(eval_times)*1000 if eval_times else 0.0
    
    stats.append(f"Avg move time: {avg_move_time:.2f}ms\n")
    stats.append(f"Avg eval time: {avg_eval_time:.2f}ms\n")
    return Panel(stats, title="Stats")

def main():
    """Self-playing game with performance metrics"""
    # Create a 4x4x4 game where you need 4 in a row to win
    config = GameConfig(size=3, target=3)
    game = TicTacToe3D(config)
    
    # Performance tracking
    move_times: deque = deque(maxlen=100)
    eval_times: deque = deque(maxlen=100)
    total_moves = 0
    total_evals = 0
    start_time = time.time()
    
    # Create layout
    layout = Layout()
    layout.split_row(
        Layout(name="board", ratio=2),
        Layout(name="stats", ratio=1)
    )
    
    with Live(layout, refresh_per_second=4) as live:
        while True:
            # Update board display
            layout["board"].update(Panel(create_board_table(game), title="3D Tic-Tac-Toe"))
            
            # Get game state and track evaluation time
            eval_start = time.time()
            state = game.get_game_state()
            eval_time = time.time() - eval_start
            eval_times.append(eval_time)
            total_evals += 1
            
            # Calculate performance metrics
            elapsed = time.time() - start_time
            moves_per_sec = total_moves / elapsed if elapsed > 0 else 0
            evals_per_sec = total_evals / elapsed if elapsed > 0 else 0
            
            # Update stats display
            layout["stats"].update(create_stats_panel(
                moves_per_sec, evals_per_sec,
                total_moves, total_evals,
                list(move_times), list(eval_times)
            ))
            
            if state != GameResult.IN_PROGRESS:
                layout["board"].update(Panel(
                    create_board_table(game), 
                    title=f"Game Over! Result: {state.name}"
                ))
                break
            
            # Get and make AI move
            move_start = time.time()
            move = game.get_best_move(depth=10)
            move_time = time.time() - move_start
            move_times.append(move_time)
            total_moves += 1
            
            if move:
                game.make_move(*move)
            else:
                layout["board"].update(Panel(
                    create_board_table(game), 
                    title="Game Over! No valid moves"
                ))
                break

if __name__ == "__main__":
    main()
