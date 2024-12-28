from hypothesis import given, strategies as st, settings, assume, example
import hypothesis.strategies as st
import pytest
import numpy as np
from tictactoe3d import (
    GameConfig,
    TicTacToe3D,
    BitBoard,
    WinPatternGenerator,
    GameResult
)
from typing import List
import json
from pathlib import Path
# Custom strategies
@st.composite
def valid_game_configs(draw):
    """Generate valid game configurations"""
    size = draw(st.integers(min_value=3, max_value=5))
    target = draw(st.integers(min_value=3, max_value=size))
    return GameConfig(size=size, target=target)

@st.composite
def valid_board_pairs(draw, config: GameConfig):
    """Generate valid board pairs for a given config"""
    board_p1 = BitBoard(config)
    board_p2 = BitBoard(config)
    
    # Generate random moves
    n_moves = draw(st.integers(min_value=0, max_value=config.size**3 // 2))
    positions = draw(st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=config.size-1),
            st.integers(min_value=0, max_value=config.size-1),
            st.integers(min_value=0, max_value=config.size-1)
        ),
        min_size=n_moves,
        max_size=n_moves,
        unique=True
    ))
    
    for x, y, z in positions[:n_moves//2]:
        board_p1.set_bit(x, y, z)
    for x, y, z in positions[n_moves//2:]:
        board_p2.set_bit(x, y, z)
    
    return board_p1, board_p2

@st.composite
def valid_game_states(draw):
    """Generate complete valid game states"""
    config = draw(valid_game_configs())
    board_p1, board_p2 = draw(valid_board_pairs(config))
    return config, board_p1, board_p2

class TestGameConfig:
    def test_valid_config(self):
        config = GameConfig(size=3, target=3)
        assert config.size == 3
        assert config.target == 3

    def test_invalid_size(self):
        with pytest.raises(AssertionError):
            GameConfig(size=2, target=3)

    def test_invalid_target(self):
        with pytest.raises(AssertionError):
            GameConfig(size=3, target=4)

class TestBitBoard:
    @pytest.fixture
    def config_3x3(self):
        return GameConfig(size=3, target=3)

    def test_initialization(self, config_3x3):
        board = BitBoard(config_3x3)
        assert board.is_empty()
        assert board.count_bits() == 0

    def test_set_get_bit(self, config_3x3):
        board = BitBoard(config_3x3)
        board.set_bit(0, 0, 0)
        assert board.get_bit(0, 0, 0)
        assert not board.get_bit(0, 0, 1)

    def test_clear_bit(self, config_3x3):
        board = BitBoard(config_3x3)
        board.set_bit(0, 0, 0)
        board.clear_bit(0, 0, 0)
        assert not board.get_bit(0, 0, 0)

    def test_bitwise_operations(self, config_3x3):
        board1 = BitBoard(config_3x3)
        board2 = BitBoard(config_3x3)
        
        board1.set_bit(0, 0, 0)
        board2.set_bit(1, 1, 1)
        
        combined = board1 | board2
        assert combined.get_bit(0, 0, 0)
        assert combined.get_bit(1, 1, 1)

class TestWinPatternGenerator:
    @pytest.fixture
    def generator_3x3(self):
        return WinPatternGenerator(GameConfig(size=3, target=3))

    def test_pattern_generation(self, generator_3x3):
        patterns = generator_3x3.generate_all_patterns()
        assert len(patterns) > 0
        
        # Test horizontal pattern
        pattern = generator_3x3.generate_pattern(0, 0, 0, 1, 0, 0)
        assert pattern is not None
        assert pattern.count_bits() == 3

    def test_invalid_pattern(self, generator_3x3):
        # Test pattern that goes out of bounds
        pattern = generator_3x3.generate_pattern(2, 2, 2, 1, 0, 0)
        assert pattern is None

class TestTicTacToe3D:
    @pytest.fixture
    def game_3x3(self):
        return TicTacToe3D(GameConfig(size=3, target=3))

    def test_initialization(self, game_3x3):
        assert game_3x3.current_player == 1
        assert game_3x3.board_p1.is_empty()
        assert game_3x3.board_p2.is_empty()

    @given(valid_game_states())
    @settings(deadline=None) 
    def test_evaluation_consistency(self, game_state):
        config, board_p1, board_p2 = game_state
        game = TicTacToe3D(config)
        
        result1 = game.evaluate_batch([board_p1], [board_p2])[0]
        result2 = game.evaluate_batch([board_p1], [board_p2])[0]
        assert result1 == result2

    def test_horizontal_win(self, game_3x3):
        # Make horizontal winning line
        game_3x3.make_move(0, 0, 0)
        game_3x3.make_move(0, 1, 0)  # Player 2
        game_3x3.make_move(1, 0, 0)
        game_3x3.make_move(1, 1, 0)  # Player 2
        game_3x3.make_move(2, 0, 0)
        
        assert game_3x3.get_game_state() == GameResult.WIN

    def test_valid_moves(self, game_3x3):
        game_3x3.make_move(0, 0, 0)
        moves = game_3x3.get_valid_moves(game_3x3.board_p1, game_3x3.board_p2)
        assert len(moves) == game_3x3.size ** 3 - 1
        assert (0, 0, 0) not in moves

    @pytest.mark.benchmark(
        group="perfect_play",
        min_rounds=1,  # This is a deep test, so one round is sufficient
        warmup=False
    )
    def test_perfect_play_draw(self, game_3x3, benchmark):
        """Verify that 3x3x3 with perfect play results in a draw"""
        def simulate_perfect_game():
            while True:
                state = game_3x3.get_game_state()
                if state != GameResult.IN_PROGRESS:
                    return state
                
                # Get best move with deep search for perfect play
                move = game_3x3.get_best_move(depth=20)
                if move is None:
                    return game_3x3.get_game_state()
                game_3x3.make_move(*move)
        
        # Run the perfect game simulation
        result = benchmark(simulate_perfect_game)
        assert result == GameResult.DRAW, "3x3x3 should be a draw with perfect play"

class TestPerformance:
    @pytest.fixture
    def sample_boards(self) -> tuple[List[BitBoard], List[BitBoard]]:
        """Generate sample boards for performance testing"""
        config = GameConfig(size=4, target=4)
        n_positions = 1024
        
        boards_p1 = []
        boards_p2 = []
        
        # Generate boards with known patterns
        for _ in range(n_positions):
            b1 = BitBoard(config)
            b2 = BitBoard(config)
            
            # Add some random moves
            for _ in range(4):
                x = np.random.randint(0, 4)
                y = np.random.randint(0, 4)
                z = np.random.randint(0, 4)
                if not (b1.get_bit(x, y, z) or b2.get_bit(x, y, z)):
                    b1.set_bit(x, y, z)
            
            boards_p1.append(b1)
            boards_p2.append(b2)
        
        return boards_p1, boards_p2

    def run_performance_test(self, benchmark, test_name: str, test_fn, 
                           test_args: dict, iterations: int,
                           improvement_threshold: float = 0.9) -> None:
        """
        Run a performance test with baseline comparison and optional baseline update
        
        Args:
            benchmark: pytest benchmark fixture
            test_name: Name of the test for storing in snapshots
            test_fn: Function to benchmark
            test_args: Arguments to pass to the test function
            iterations: Number of iterations being tested
            improvement_threshold: Threshold for updating baseline (e.g., 0.9 means 10% faster)
        """
        # Run benchmark
        result = benchmark(test_fn)
        
        # Store performance metrics
        snapshot_file = Path("performance_snapshots.json")
        current_stats = {
            "mean": float(benchmark.stats["mean"]),
            "stddev": float(benchmark.stats["stddev"]),
            "rounds": int(benchmark.stats["rounds"]),
            "iterations": iterations
        }
        
        if snapshot_file.exists():
            data = json.loads(snapshot_file.read_text())
        else:
            data = {}
        
        if test_name not in data:
            print(f"\nFirst run - storing baseline {test_name} metrics:")
            print(f"Mean time: {current_stats['mean']:.6f} seconds")
            print(f"Std dev:   {current_stats['stddev']:.6f} seconds")
            data[test_name] = current_stats
            snapshot_file.write_text(json.dumps(data, indent=2))
        else:
            baseline = data[test_name]
            
            # Compare with baseline
            percent_change = ((current_stats["mean"] - baseline["mean"]) / baseline["mean"]) * 100
            
            print(f"\n{test_name} Performance comparison:")
            print(f"Baseline mean: {baseline['mean']:.6f} seconds")
            print(f"Current mean:  {current_stats['mean']:.6f} seconds")
            print(f"Change:        {percent_change:+.2f}%")
            
            # Update baseline if there's a significant improvement
            if current_stats["mean"] <= baseline["mean"] * improvement_threshold:
                print(f"\nSignificant improvement detected! Updating baseline.")
                data[test_name] = current_stats
                snapshot_file.write_text(json.dumps(data, indent=2))
            # Still fail if it's a significant regression
            elif current_stats["mean"] > baseline["mean"] * 1.2:  # 20% regression threshold
                assert False, f"Performance regression detected: {percent_change:.1f}% slower than baseline"
        
        return result

    @pytest.mark.benchmark(
        group="cuda",
        min_rounds=10,
        warmup=True,
        warmup_iterations=5
    )
    def test_batch_evaluation_performance(self, benchmark, sample_boards):
        """Benchmark batch evaluation performance with regression detection"""
        boards_p1, boards_p2 = sample_boards
        game = TicTacToe3D(GameConfig(size=4, target=4))
        
        def run_batch():
            return game.evaluate_batch(boards_p1, boards_p2)
        
        # Run performance test
        eval_result = self.run_performance_test(
            benchmark=benchmark,
            test_name="batch_evaluation",
            test_fn=run_batch,
            test_args={},
            iterations=len(boards_p1),
            improvement_threshold=0.9  # Update baseline if 10% faster
        )
        
        # Verify correctness
        assert len(eval_result) == len(boards_p1)
        assert all(isinstance(r, np.int32) for r in eval_result)

    @pytest.mark.benchmark(
        group="minimax",
        min_rounds=5,  # Fewer rounds since move generation is slower
        warmup=True,
        warmup_iterations=2
    )
    def test_move_generation_performance(self, benchmark, sample_boards):
        """Benchmark move generation performance with regression detection"""
        boards_p1, boards_p2 = sample_boards
        game = TicTacToe3D(GameConfig(size=4, target=4))
        
        # Take a subset of boards for move generation since it's slower
        test_positions = 10
        boards_p1 = boards_p1[:test_positions]
        boards_p2 = boards_p2[:test_positions]
        
        def run_move_generation():
            moves = []
            for b1, b2 in zip(boards_p1, boards_p2):
                game.board_p1 = b1
                game.board_p2 = b2
                moves.append(game.get_best_move(depth=5))
            return moves
        
        # Run performance test
        move_result = self.run_performance_test(
            benchmark=benchmark,
            test_name="move_generation",
            test_fn=run_move_generation,
            test_args={},
            iterations=test_positions,
            improvement_threshold=0.9  # Update baseline if 10% faster
        )
        
        # Verify correctness
        assert len(move_result) == test_positions
        # Each result should be None or a valid move tuple
        for move in move_result:
            if move is not None:
                assert len(move) == 3
                assert all(isinstance(x, int) for x in move)
                assert all(0 <= x < 4 for x in move)

if __name__ == "__main__":
    pytest.main([__file__])
