"""
3D Tic-Tac-Toe with CUDA acceleration
Supports arbitrary NxNxN board sizes with efficient bit-based representation
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Set, Optional, Tuple, Union, Protocol, runtime_checkable
from abc import ABC, abstractmethod
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
import argparse
import cProfile
import pstats
from pstats import SortKey

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
def evaluate_position_kernel(board_p1, board_p2, result, patterns,
                           n_patterns: int32, n_u64s: int32): # type: ignore
    """CUDA kernel for single position evaluation"""
    if cuda.grid(1) > 0:  # Only use first thread
        return
    
    # Check each pattern
    for pattern_idx in range(n_patterns):
        pattern_offset = pattern_idx * n_u64s
        
        # Check player 1 win
        if check_pattern_match(
            board_p1, 
            patterns[pattern_offset:pattern_offset + n_u64s],
            n_u64s
        ):
            result[0] = GameResult.WIN.value
            return
        
        # Check player 2 win
        if check_pattern_match(
            board_p2,
            patterns[pattern_offset:pattern_offset + n_u64s],
            n_u64s
        ):
            result[0] = GameResult.LOSS.value
            return
    
    # Check for draw
    is_full = True
    for u64_idx in range(n_u64s):
        if (board_p1[u64_idx] | board_p2[u64_idx]) != ~np.uint64(0):
            is_full = False
            break
    
    result[0] = GameResult.DRAW.value if is_full else GameResult.IN_PROGRESS.value

@cuda.jit
def evaluate_boards_kernel(boards_p1, boards_p2, results, patterns, 
                         n_patterns: int32, n_u64s: int32): # type: ignore
    """CUDA kernel for batch position evaluation"""
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

@cuda.jit(device=True)
def get_valid_moves_device(board_p1: uint64[:], board_p2: uint64[:],
                          moves: int32[:], n_moves: int32[:],
                          size: int32, num_u64s: int32) -> None:
    """Device function to get valid moves"""
    n_moves[0] = 0
    total_positions = size * size * size
    
    for pos in range(total_positions):
        array_idx = pos // 64
        bit_idx = pos % 64
        mask = np.uint64(1) << bit_idx
        
        if not ((board_p1[array_idx] & mask) or (board_p2[array_idx] & mask)):
            idx = cuda.atomic.add(n_moves, 0, 1)
            if idx < moves.shape[0]:
                moves[idx] = pos

@cuda.jit(device=True)
def make_move_device(board: uint64[:], pos: int32, size: int32) -> None:
    """Device function to make a move"""
    array_idx = pos // 64
    bit_idx = pos % 64
    board[array_idx] |= np.uint64(1) << bit_idx

@cuda.jit(device=True)
def clear_move_device(board: uint64[:], pos: int32, size: int32) -> None:
    """Device function to clear a move"""
    array_idx = pos // 64
    bit_idx = pos % 64
    board[array_idx] &= ~(np.uint64(1) << bit_idx)

@cuda.jit(device=True)
def evaluate_heuristic_device(board_p1: uint64[:], board_p2: uint64[:],
                            patterns: uint64[:], n_patterns: int32,
                            num_u64s: int32) -> float32:
    """Device function for heuristic evaluation"""
    score = float32(0.0)
    total_patterns = float32(n_patterns)
    
    for pattern_idx in range(n_patterns):
        pattern_offset = pattern_idx * num_u64s
        p1_match = float32(0.0)
        p2_match = float32(0.0)
        
        for i in range(num_u64s):
            pattern_bits = patterns[pattern_offset + i]
            p1_match += float32(cuda.popc(board_p1[i] & pattern_bits))
            p2_match += float32(cuda.popc(board_p2[i] & pattern_bits))
        
        score += (p1_match - p2_match) / total_patterns
    
    return score

@cuda.jit(device=True)
def minimax_device(board_p1: uint64[:], board_p2: uint64[:],
                  patterns: uint64[:], n_patterns: int32,
                  depth: int32, alpha: float32, beta: float32,
                  maximizing: bool, size: int32, num_u64s: int32,
                  best_move: int32[:], temp_board: uint64[:],
                  moves: int32[:], n_moves: int32[:]) -> float32:
    """Device implementation of minimax algorithm"""
    # Check terminal states
    result = float32(0.0)
    is_terminal = False
    
    # Check each pattern for wins
    for pattern_idx in range(n_patterns):
        pattern_offset = pattern_idx * num_u64s
        
        # Check player 1 win
        if check_pattern_match(board_p1, patterns[pattern_offset:pattern_offset + num_u64s], num_u64s):
            return float32(1.0)
        
        # Check player 2 win
        if check_pattern_match(board_p2, patterns[pattern_offset:pattern_offset + num_u64s], num_u64s):
            return float32(-1.0)
    
    # Check for draw
    is_full = True
    for i in range(num_u64s):
        if (board_p1[i] | board_p2[i]) != ~np.uint64(0):
            is_full = False
            break
    
    if is_full:
        return float32(0.0)
    elif depth == 0:
        return evaluate_heuristic_device(board_p1, board_p2, patterns, n_patterns, num_u64s)
    
    # Get valid moves
    get_valid_moves_device(board_p1, board_p2, moves, n_moves, size, num_u64s)
    if n_moves[0] == 0:
        return float32(0.0)
    
    # Copy current board state to temp board
    for i in range(num_u64s):
        temp_board[i] = board_p1[i] if maximizing else board_p2[i]
    
    if maximizing:
        max_eval = float32(-1000.0)
        for i in range(n_moves[0]):
            move = moves[i]
            make_move_device(temp_board, move, size)
            
            eval = minimax_device(temp_board, board_p2, patterns, n_patterns,
                                depth - 1, alpha, beta, False,
                                size, num_u64s, best_move, temp_board,
                                moves, n_moves)
            
            clear_move_device(temp_board, move, size)
            
            if eval > max_eval:
                max_eval = eval
                if depth == best_move[1]:  # If at root
                    best_move[0] = move
            
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        
        return max_eval
    else:
        min_eval = float32(1000.0)
        for i in range(n_moves[0]):
            move = moves[i]
            make_move_device(temp_board, move, size)
            
            eval = minimax_device(board_p1, temp_board, patterns, n_patterns,
                                depth - 1, alpha, beta, True,
                                size, num_u64s, best_move, temp_board,
                                moves, n_moves)
            
            clear_move_device(temp_board, move, size)
            
            if eval < min_eval:
                min_eval = eval
                if depth == best_move[1]:  # If at root
                    best_move[0] = move
            
            beta = min(beta, eval)
            if beta <= alpha:
                break
        
        return min_eval

@cuda.jit(device=True)
def parallel_minimax_device(board_p1: uint64[:], board_p2: uint64[:],
                          patterns: uint64[:], n_patterns: int32,
                          depth: int32, alpha: float32, beta: float32,
                          maximizing: bool, size: int32, num_u64s: int32,
                          best_move: int32[:], temp_board: uint64[:],
                          moves: int32[:], n_moves: int32[:],
                          beam_width: int32 = 8) -> float32:
    """Parallel minimax with beam search optimization"""
    # Terminal state checks using warp-level parallelism
    warp_id = cuda.threadIdx.x & 0x1f
    if warp_id < n_patterns:
        pattern_offset = warp_id * num_u64s
        if check_pattern_match(board_p1, patterns[pattern_offset:pattern_offset + num_u64s], num_u64s):
            return float32(1.0)
        if check_pattern_match(board_p2, patterns[pattern_offset:pattern_offset + num_u64s], num_u64s):
            return float32(-1.0)
    
    if depth == 0:
        return evaluate_heuristic_device(board_p1, board_p2, patterns, n_patterns, num_u64s)
    
    # Get valid moves
    get_valid_moves_device(board_p1, board_p2, moves, n_moves, size, num_u64s)
    if n_moves[0] == 0:
        return float32(0.0)
    
    # Use beam search to evaluate only the most promising moves
    n_eval = min(n_moves[0], beam_width)
    
    if maximizing:
        max_eval = float32(-1000.0)
        for i in range(n_eval):
            move = moves[i]
            make_move_device(temp_board, move, size)
            
            eval = parallel_minimax_device(temp_board, board_p2, patterns, n_patterns,
                                        depth - 1, alpha, beta, False,
                                        size, num_u64s, best_move, temp_board,
                                        moves, n_moves, beam_width)
            
            clear_move_device(temp_board, move, size)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float32(1000.0)
        for i in range(n_eval):
            move = moves[i]
            make_move_device(temp_board, move, size)
            
            eval = parallel_minimax_device(board_p1, temp_board, patterns, n_patterns,
                                        depth - 1, alpha, beta, True,
                                        size, num_u64s, best_move, temp_board,
                                        moves, n_moves, beam_width)
            
            clear_move_device(temp_board, move, size)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

@cuda.jit
def find_best_move_kernel(board_p1: uint64[:], board_p2: uint64[:],
                         patterns: uint64[:], best_move: int32[:],
                         depth: int32, size: int32, num_u64s: int32,
                         n_patterns: int32):
    """Parallel kernel using block-level evaluation and warp-level pattern matching"""
    # Get block and thread indices
    block_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x
    warp_idx = thread_idx // WARP_SIZE
    lane_idx = thread_idx % WARP_SIZE
    
    # Shared memory for block-wide computation
    shared_board = cuda.shared.array(shape=64, dtype=uint64)
    shared_moves = cuda.shared.array(shape=512, dtype=int32)
    shared_n_moves = cuda.shared.array(shape=1, dtype=int32)
    shared_evals = cuda.shared.array(shape=32, dtype=float32)  # One per warp
    shared_patterns = cuda.shared.array(shape=64, dtype=uint64)  # Cache patterns
    
    # First warp initializes shared memory
    if warp_idx == 0:
        if lane_idx < num_u64s:
            shared_board[lane_idx] = board_p1[lane_idx]
        if lane_idx == 0:
            shared_n_moves[0] = 0
            get_valid_moves_device(board_p1, board_p2, shared_moves, shared_n_moves, size, num_u64s)
    cuda.syncthreads()
    
    n_moves = shared_n_moves[0]
    moves_per_block = (n_moves + cuda.gridDim.x - 1) // cuda.gridDim.x
    start_move = block_idx * moves_per_block
    end_move = min(start_move + moves_per_block, n_moves)
    
    # Each warp handles a subset of moves
    moves_per_warp = (end_move - start_move + (cuda.blockDim.x // WARP_SIZE) - 1) // (cuda.blockDim.x // WARP_SIZE)
    my_start = start_move + warp_idx * moves_per_warp
    my_end = min(my_start + moves_per_warp, end_move)
    
    # Warp-level evaluation
    if my_start < my_end:
        warp_best_eval = float32(-1000.0)
        warp_best_move = int32(-1)
        
        # Each thread in warp helps evaluate position
        for move_idx in range(my_start, my_end):
            move = shared_moves[move_idx]
            
            # Make move cooperatively within warp
            if lane_idx < num_u64s:
                make_move_device(shared_board, move, size)
            cuda.syncwarp()
            
            # Evaluate position using warp-level pattern matching
            eval = float32(0.0)
            patterns_per_thread = (n_patterns + WARP_SIZE - 1) // WARP_SIZE
            for i in range(patterns_per_thread):
                pattern_idx = i * WARP_SIZE + lane_idx
                if pattern_idx < n_patterns:
                    pattern_offset = pattern_idx * num_u64s
                    if check_pattern_match(shared_board, patterns[pattern_offset:pattern_offset + num_u64s], num_u64s):
                        eval = float32(1.0)
                        break
            
            # Warp reduction to get final evaluation
            for offset in [16, 8, 4, 2, 1]:
                other = cuda.shfl_xor_sync(0xffffffff, eval, offset)
                eval = max(eval, other)
            
            if lane_idx == 0:
                if eval > warp_best_eval:
                    warp_best_eval = eval
                    warp_best_move = move
            
            # Undo move cooperatively
            if lane_idx < num_u64s:
                clear_move_device(shared_board, move, size)
            cuda.syncwarp()
        
        # Store warp results
        if lane_idx == 0:
            shared_evals[warp_idx] = warp_best_eval
            if warp_best_move >= 0:
                cuda.atomic.max(best_move, 0, warp_best_move)
    
    cuda.syncthreads()
    
    # Final reduction across warps (first warp only)
    if warp_idx == 0 and lane_idx == 0:
        block_best_eval = float32(-1000.0)
        n_warps = (cuda.blockDim.x + WARP_SIZE - 1) // WARP_SIZE
        for i in range(n_warps):
            if shared_evals[i] > block_best_eval:
                block_best_eval = shared_evals[i]

class TicTacToeEngine(ABC):
    """Abstract base class for TicTacToe engines"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name"""
        pass
    
    @abstractmethod
    def evaluate_position(self, board_p1: BitBoard, board_p2: BitBoard) -> GameResult:
        """Evaluate current position"""
        pass
    
    @abstractmethod
    def get_best_move(self, board_p1: BitBoard, board_p2: BitBoard, depth: int) -> Optional[Tuple[int, int, int]]:
        """Get best move for current position"""
        pass

class CPUEngine(TicTacToeEngine):
    """Pure CPU-based engine using numpy"""
    name = "CPU Engine"
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.size
        self.num_u64s = config.num_u64s
        
        # Generate winning patterns
        generator = WinPatternGenerator(config)
        self.patterns = generator.generate_all_patterns()
        
        # Convert patterns to numpy array for efficient operations
        self.pattern_array = np.zeros((len(self.patterns), self.num_u64s), dtype=np.uint64)
        for i, pattern in enumerate(self.patterns):
            self.pattern_array[i] = pattern.bits
    
    def evaluate_position(self, board_p1: BitBoard, board_p2: BitBoard) -> GameResult:
        """Evaluate current position using numpy operations"""
        # Check for wins using vectorized operations
        p1_bits = board_p1.bits
        p2_bits = board_p2.bits
        
        # Check player 1 win
        for pattern_bits in self.pattern_array:
            if np.all((p1_bits & pattern_bits) == pattern_bits):
                return GameResult.WIN
        
        # Check player 2 win
        for pattern_bits in self.pattern_array:
            if np.all((p2_bits & pattern_bits) == pattern_bits):
                return GameResult.LOSS
        
        # Check for draw
        if np.all((p1_bits | p2_bits) == ~np.uint64(0)):
            return GameResult.DRAW
        
        return GameResult.IN_PROGRESS
    
    def get_valid_moves(self, board_p1: BitBoard, board_p2: BitBoard) -> List[Tuple[int, int, int]]:
        """Get list of valid moves using numpy operations"""
        moves = []
        combined_bits = board_p1.bits | board_p2.bits
        
        # Use numpy to find empty positions
        for z in range(self.size):
            for y in range(self.size):
                for x in range(self.size):
                    pos = x + y * self.size + z * self.size * self.size
                    array_idx, bit_idx = pos // 64, pos % 64
                    if not (combined_bits[array_idx] & (np.uint64(1) << bit_idx)):
                        moves.append((x, y, z))
        
        return moves
    
    def minimax(self, board_p1: BitBoard, board_p2: BitBoard, depth: int,
                alpha: float = float('-inf'), beta: float = float('inf'),
                maximizing: bool = True) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """Pure CPU minimax implementation"""
        result = self.evaluate_position(board_p1, board_p2)
        
        if result == GameResult.WIN:
            return 1.0, None
        elif result == GameResult.LOSS:
            return -1.0, None
        elif result == GameResult.DRAW:
            return 0.0, None
        elif depth == 0:
            # Simple heuristic based on pattern matching
            score = 0.0
            total_patterns = len(self.patterns)
            
            for pattern_bits in self.pattern_array:
                p1_match = np.sum(np.bitwise_and(board_p1.bits, pattern_bits))
                p2_match = np.sum(np.bitwise_and(board_p2.bits, pattern_bits))
                score += (p1_match - p2_match) / total_patterns
            
            return score, None
        
        moves = self.get_valid_moves(board_p1, board_p2)
        if not moves:
            return 0.0, None
        
        # Create reusable BitBoard objects
        temp_board = BitBoard(self.config)
        best_move = None
        
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                temp_board.bits[:] = board_p1.bits
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
                temp_board.bits[:] = board_p2.bits
                temp_board.set_bit(*move)
                
                eval, _ = self.minimax(board_p1, temp_board, depth-1, alpha, beta, True)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
    
    def get_best_move(self, board_p1: BitBoard, board_p2: BitBoard, depth: int) -> Optional[Tuple[int, int, int]]:
        """Get best move using CPU-based minimax"""
        _, move = self.minimax(board_p1, board_p2, depth)
        return move

class GPUEngine(TicTacToeEngine):
    """Pure GPU-based engine using CUDA"""
    name = "GPU Engine"
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.size
        self.num_u64s = config.num_u64s
        
        # Generate winning patterns
        generator = WinPatternGenerator(config)
        self.patterns = generator.generate_all_patterns()
        
        # Convert patterns to CUDA-friendly format and transfer to GPU
        pattern_array = np.zeros((len(self.patterns) * self.num_u64s), dtype=np.uint64)
        for i, pattern in enumerate(self.patterns):
            pattern_array[i * self.num_u64s:(i + 1) * self.num_u64s] = pattern.bits
        
        # Create pinned memory and transfer patterns to GPU
        self.pattern_array = cuda.pinned_array(pattern_array.shape, dtype=np.uint64)
        self.pattern_array[:] = pattern_array
        self.d_patterns = cuda.to_device(self.pattern_array)
        
        # Calculate optimal thread configuration
        device = cuda.get_current_device()
        self.max_threads_per_block = min(device.MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK)
        self.warp_size = WARP_SIZE
        self.threadsperblock = min(
            self.max_threads_per_block,
            ((self.max_threads_per_block // self.warp_size) * self.warp_size)
        )
        
    def evaluate_position(self, board_p1: BitBoard, board_p2: BitBoard) -> GameResult:
        """Evaluate position using GPU"""
        # Transfer boards to device
        d_board_p1 = cuda.to_device(board_p1.bits)
        d_board_p2 = cuda.to_device(board_p2.bits)
        d_result = cuda.device_array(1, dtype=np.int32)
        
        # Launch kernel
        evaluate_position_kernel[1, 1](
            d_board_p1, d_board_p2, d_result,
            self.d_patterns, np.int32(len(self.patterns)), np.int32(self.num_u64s)
        )
        
        return GameResult(d_result.copy_to_host()[0])
    
    def get_best_move(self, board_p1: BitBoard, board_p2: BitBoard, depth: int) -> Optional[Tuple[int, int, int]]:
        """Get best move using GPU-accelerated parallel minimax"""
        # Allocate device memory for result
        best_move = cuda.device_array(2, dtype=np.int32)  # [move, original_depth]
        
        # Calculate optimal grid and block dimensions
        device = cuda.get_current_device()
        max_threads = min(device.MAX_THREADS_PER_BLOCK, 256)  # Use smaller thread blocks for better occupancy
        threads_per_block = min(max_threads, ((max_threads // WARP_SIZE) * WARP_SIZE))
        
        # Estimate number of moves for grid size
        total_positions = self.size * self.size * self.size
        max_moves = total_positions  # Upper bound
        min_blocks = (max_moves + threads_per_block - 1) // threads_per_block
        max_blocks = device.MULTIPROCESSOR_COUNT * MIN_BLOCKS_PER_SM
        blocks_per_grid = max(min_blocks, max_blocks)
        
        # Launch kernel with optimal configuration
        find_best_move_kernel[blocks_per_grid, threads_per_block](
            cuda.to_device(board_p1.bits),
            cuda.to_device(board_p2.bits),
            self.d_patterns,
            best_move,
            np.int32(depth),
            np.int32(self.size),
            np.int32(self.num_u64s),
            np.int32(len(self.patterns))
        )
        
        # Get result
        move_pos = best_move[0].copy_to_host()
        if move_pos < 0:
            return None
        
        # Convert position to coordinates
        x = move_pos % self.size
        y = (move_pos // self.size) % self.size
        z = move_pos // (self.size * self.size)
        return (x, y, z)

class MixedEngine(TicTacToeEngine):
    """Hybrid engine using both CPU and GPU"""
    name = "Mixed Engine"
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.size = config.size
        self.num_u64s = config.num_u64s
        
        # Initialize both CPU and GPU components
        self.cpu_engine = CPUEngine(config)
        self.gpu_engine = GPUEngine(config)
        
        # Use GPU for pattern matching and position evaluation
        self.patterns = self.gpu_engine.patterns
        self.d_patterns = self.gpu_engine.d_patterns
        
        # Preallocate pinned memory for batch evaluation
        self.max_batch_size = 1024
        self.h_boards_p1 = cuda.pinned_array((self.max_batch_size, self.num_u64s), dtype=np.uint64)
        self.h_boards_p2 = cuda.pinned_array((self.max_batch_size, self.num_u64s), dtype=np.uint64)
    
    def evaluate_batch(self, boards_p1: List[BitBoard], boards_p2: List[BitBoard]) -> npt.NDArray[np.int32]:
        """Evaluate multiple positions in parallel using GPU"""
        n_boards = len(boards_p1)
        assert len(boards_p2) == n_boards, "Must provide equal number of boards"
        assert n_boards <= self.max_batch_size, f"Batch size {n_boards} exceeds maximum {self.max_batch_size}"
        
        # Copy boards to pinned memory
        for i in range(n_boards):
            self.h_boards_p1[i] = boards_p1[i].bits
            self.h_boards_p2[i] = boards_p2[i].bits
        
        # Transfer to device
        d_boards_p1 = cuda.to_device(self.h_boards_p1[:n_boards])
        d_boards_p2 = cuda.to_device(self.h_boards_p2[:n_boards])
        d_results = cuda.device_array(n_boards, dtype=np.int32)
        
        # Calculate grid size
        min_grid_size = (n_boards + self.gpu_engine.threadsperblock - 1) // self.gpu_engine.threadsperblock
        device = cuda.get_current_device()
        max_blocks = device.MULTIPROCESSOR_COUNT * MIN_BLOCKS_PER_SM
        blockspergrid = max(min_grid_size, max_blocks)
        
        # Launch kernel
        evaluate_boards_kernel[blockspergrid, self.gpu_engine.threadsperblock](
            d_boards_p1, d_boards_p2, d_results,
            self.d_patterns, np.int32(len(self.patterns)), np.int32(self.num_u64s)
        )
        
        return d_results.copy_to_host()

    def evaluate_position(self, board_p1: BitBoard, board_p2: BitBoard) -> GameResult:
        """Evaluate single position using GPU"""
        return self.gpu_engine.evaluate_position(board_p1, board_p2)

    def get_valid_moves(self, board_p1: BitBoard, board_p2: BitBoard) -> List[Tuple[int, int, int]]:
        """Get valid moves using CPU"""
        return self.cpu_engine.get_valid_moves(board_p1, board_p2)
    
    def minimax(self, board_p1: BitBoard, board_p2: BitBoard, depth: int,
                alpha: float = float('-inf'), beta: float = float('inf'),
                maximizing: bool = True) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """Hybrid minimax using GPU for evaluation and CPU for search"""
        result = self.evaluate_position(board_p1, board_p2)
        
        if result == GameResult.WIN:
            return 1.0, None
        elif result == GameResult.LOSS:
            return -1.0, None
        elif result == GameResult.DRAW:
            return 0.0, None
        elif depth == 0:
            # Use GPU-accelerated batch evaluation for heuristic
            boards_p1 = [board_p1] * len(self.patterns)
            boards_p2 = [board_p2] * len(self.patterns)
            pattern_scores = self.evaluate_batch(boards_p1, boards_p2)
            return float(np.mean(pattern_scores)), None
        
        moves = self.get_valid_moves(board_p1, board_p2)
        if not moves:
            return 0.0, None
        
        # Create reusable BitBoard objects
        temp_board = BitBoard(self.config)
        best_move = None
        
        # Simple move ordering heuristic
        mid = self.size // 2
        moves.sort(key=lambda m: -(abs(m[0] - mid) + abs(m[1] - mid) + abs(m[2] - mid)))
        
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                temp_board.bits[:] = board_p1.bits
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
                temp_board.bits[:] = board_p2.bits
                temp_board.set_bit(*move)
                
                eval, _ = self.minimax(board_p1, temp_board, depth-1, alpha, beta, True)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
    
    def get_best_move(self, board_p1: BitBoard, board_p2: BitBoard, depth: int) -> Optional[Tuple[int, int, int]]:
        """Get best move using hybrid approach"""
        _, move = self.minimax(board_p1, board_p2, depth)
        return move

class TicTacToe3D:
    """Main game class for arbitrary-sized 3D Tic-Tac-Toe"""
    
    def __init__(self, config: GameConfig, engine_type: str = "mixed"):
        self.config = config
        self.size = config.size
        self.num_u64s = config.num_u64s
        
        # Initialize game state
        self.board_p1 = BitBoard(config)
        self.board_p2 = BitBoard(config)
        self.current_player = 1
        
        # Create engine based on type
        if engine_type.lower() == "cpu":
            self._engine: TicTacToeEngine = CPUEngine(config)
        elif engine_type.lower() == "gpu":
            self._engine = GPUEngine(config)
        else:
            self._engine = MixedEngine(config)
    
    @property
    def engine(self) -> TicTacToeEngine:
        """Get the current engine"""
        return self._engine
    
    def get_game_state(self) -> GameResult:
        """Get current game state"""
        return self.engine.evaluate_position(self.board_p1, self.board_p2)
    
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
    
    def get_best_move(self, depth: int = 4) -> Optional[Tuple[int, int, int]]:
        """Get best move using selected engine"""
        return self.engine.get_best_move(self.board_p1, self.board_p2, depth)
    
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D Tic-Tac-Toe with optional profiling')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--depth', type=int, default=4, help='Search depth for AI (default: 4)')
    parser.add_argument('--moves', type=int, default=None, help='Number of moves to profile (default: unlimited)')
    parser.add_argument('--size', type=int, default=3, help='Board size (default: 3)')
    parser.add_argument('--target', type=int, default=3, help='Target in a row to win (default: 3)')
    parser.add_argument('--engine', type=str, choices=['cpu', 'gpu', 'mixed'], default='mixed',
                       help='Engine type to use (default: mixed)')
    parser.add_argument('--benchmark', action='store_true', help='Run engine benchmarks')
    args = parser.parse_args()

    # Create game instance
    config = GameConfig(size=args.size, target=args.target)
    
    if args.benchmark:
        # Run benchmarks for all engines
        engines = ['cpu', 'gpu', 'mixed']
        results = {}
        
        print("\nRunning engine benchmarks...")
        print("=" * 50)
        
        for engine_type in engines:
            print(f"\nTesting {engine_type.upper()} engine:")
            game = TicTacToe3D(config, engine_type=engine_type)
            
            # Measure move generation time
            move_times = []
            eval_times = []
            
            # Warm up
            for _ in range(3):
                game.get_best_move(depth=args.depth)
            
            # Actual benchmark
            n_trials = 5
            for i in range(n_trials):
                # Measure move generation
                move_start = time.time()
                move = game.get_best_move(depth=args.depth)
                move_time = time.time() - move_start
                move_times.append(move_time)
                
                if move:
                    game.make_move(*move)
                
                # Measure position evaluation
                eval_start = time.time()
                game.get_game_state()
                eval_time = time.time() - eval_start
                eval_times.append(eval_time)
            
            # Store results
            results[engine_type] = {
                'move_time': mean(move_times),
                'eval_time': mean(eval_times)
            }
            
            print(f"Average move time: {results[engine_type]['move_time']*1000:.2f}ms")
            print(f"Average eval time: {results[engine_type]['eval_time']*1000:.2f}ms")
        
        # Print comparison
        print("\nEngine Comparison:")
        print("=" * 50)
        fastest_move = min(results.items(), key=lambda x: x[1]['move_time'])[0]
        fastest_eval = min(results.items(), key=lambda x: x[1]['eval_time'])[0]
        
        print("\nMove Generation Speed:")
        baseline_move = results[fastest_move]['move_time']
        for engine, data in results.items():
            speedup = data['move_time'] / baseline_move
            print(f"{engine.upper():6s}: {data['move_time']*1000:8.2f}ms ({speedup:6.2f}x)")
        
        print("\nPosition Evaluation Speed:")
        baseline_eval = results[fastest_eval]['eval_time']
        for engine, data in results.items():
            speedup = data['eval_time'] / baseline_eval
            print(f"{engine.upper():6s}: {data['eval_time']*1000:8.2f}ms ({speedup:6.2f}x)")
        
        return
    
    # Regular game play
    game = TicTacToe3D(config, engine_type=args.engine)
    
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
    
    # Set up profiling if enabled
    profiler = cProfile.Profile() if args.profile else None
    if args.profile:
        profiler.enable()
    
    with Live(layout, refresh_per_second=4) as live:
        while True:
            # Check move limit for profiling
            if args.moves is not None and total_moves >= args.moves:
                break
                
            # Update board display
            layout["board"].update(Panel(create_board_table(game), title=f"3D Tic-Tac-Toe ({game.engine.name})"))
            
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
            move = game.get_best_move(depth=args.depth)
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
    
    # Save profiling results if enabled
    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.TIME)
        
        # Save raw profiling data
        stats.dump_stats("game_profile.prof")
        
        # Print summary to console
        print("\nTop 20 time-consuming functions:")
        stats.sort_stats(SortKey.TIME).print_stats(20)
        
        print("\nProfile data saved to game_profile.prof")
        print("To generate a flamegraph, you can use tools like:")
        print("  - flameprof game_profile.prof > profile_flame.svg")
        print("  - snakeviz game_profile.prof")

if __name__ == "__main__":
    main()
