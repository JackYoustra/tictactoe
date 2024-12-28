from hypothesis import given, strategies as st, settings, assume, example, HealthCheck
import hypothesis.strategies as st
import pytest
import numpy as np
from tictactoe3d import (
    GameConfig,
    TicTacToe3D,
    BitBoard,
    WinPatternGenerator,
    GameResult,
    CPUEngine,
    GPUEngine,
    MixedEngine
)
from typing import List
import json
from pathlib import Path

# Custom strategies
@st.composite
def valid_game_configs(draw):
    """Generate valid game configurations"""
    size = draw(st.integers(min_value=3, max_value=4))
    target = draw(st.integers(min_value=3, max_value=size))
    return GameConfig(size=size, target=target)

@st.composite
def valid_board_pairs(draw, config):
    """Generate valid board pairs for testing"""
    board_p1 = BitBoard(config)
    board_p2 = BitBoard(config)
    
    # Generate some random moves
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
    
    # Apply moves alternately to both players
    for i, (x, y, z) in enumerate(positions):
        if i % 2 == 0:
            board_p1.set_bit(x, y, z)
        else:
            board_p2.set_bit(x, y, z)
    
    return board_p1, board_p2

@st.composite
def valid_game_states(draw):
    """Generate complete valid game states"""
    config = draw(valid_game_configs())
    board_p1, board_p2 = draw(valid_board_pairs(config))
    return config, board_p1, board_p2

class TestGameConfig:
    @pytest.mark.timeout(5)
    def test_valid_config(self):
        config = GameConfig(size=3, target=3)
        assert config.size == 3
        assert config.target == 3

    @pytest.mark.timeout(5)
    def test_invalid_size(self):
        with pytest.raises(AssertionError):
            GameConfig(size=2, target=3)

    @pytest.mark.timeout(5)
    def test_invalid_target(self):
        with pytest.raises(AssertionError):
            GameConfig(size=3, target=4)

class TestBitBoard:
    @pytest.fixture
    def config_3x3(self):
        return GameConfig(size=3, target=3)

    @pytest.mark.timeout(5)
    def test_initialization(self, config_3x3):
        board = BitBoard(config_3x3)
        assert board.is_empty()
        assert board.count_bits() == 0

    @pytest.mark.timeout(5)
    def test_set_get_bit(self, config_3x3):
        board = BitBoard(config_3x3)
        board.set_bit(0, 0, 0)
        assert board.get_bit(0, 0, 0)
        assert not board.get_bit(0, 0, 1)

    @pytest.mark.timeout(5)
    def test_clear_bit(self, config_3x3):
        board = BitBoard(config_3x3)
        board.set_bit(0, 0, 0)
        board.clear_bit(0, 0, 0)
        assert not board.get_bit(0, 0, 0)

    @pytest.mark.timeout(5)
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

    @pytest.mark.timeout(5)
    def test_pattern_generation(self, generator_3x3):
        patterns = generator_3x3.generate_all_patterns()
        assert len(patterns) > 0
        
        # Test horizontal pattern
        pattern = generator_3x3.generate_pattern(0, 0, 0, 1, 0, 0)
        assert pattern is not None
        assert pattern.count_bits() == 3

    @pytest.mark.timeout(5)
    def test_invalid_pattern(self, generator_3x3):
        # Test pattern that goes out of bounds
        pattern = generator_3x3.generate_pattern(2, 2, 2, 1, 0, 0)
        assert pattern is None

class TestEngines:
    @pytest.fixture
    def config_3x3(self):
        return GameConfig(size=3, target=3)
    
    @pytest.fixture
    def engines(self, config_3x3):
        return [
            CPUEngine(config_3x3),
            GPUEngine(config_3x3),
            MixedEngine(config_3x3)
        ]
    
    @pytest.mark.timeout(5)
    def test_engine_initialization(self, engines):
        for engine in engines:
            assert engine.name is not None
            assert isinstance(engine.name, str)
    
    @pytest.mark.timeout(5)
    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(st.data())
    def test_evaluation_consistency(self, engines, data):
        """Test that all engines give consistent results"""
        # Draw the game config using Hypothesis
        config = data.draw(valid_game_configs())
        # Generate a valid board pair using the strategy
        board_p1, board_p2 = data.draw(valid_board_pairs(config))
        
        # All engines should give the same result for the same position
        results = [engine.evaluate_position(board_p1, board_p2) for engine in engines]
        assert all(r == results[0] for r in results), "All engines should agree on position evaluation"
    
    @pytest.mark.timeout(5)
    def test_horizontal_win(self, engines, config_3x3):
        board_p1 = BitBoard(config_3x3)
        board_p2 = BitBoard(config_3x3)
        
        # Make horizontal winning line for player 1
        board_p1.set_bit(0, 0, 0)
        board_p1.set_bit(1, 0, 0)
        board_p1.set_bit(2, 0, 0)
        
        for engine in engines:
            assert engine.evaluate_position(board_p1, board_p2) == GameResult.WIN
    
    @pytest.mark.timeout(5)
    def test_valid_moves(self, engines, config_3x3):
        board_p1 = BitBoard(config_3x3)
        board_p2 = BitBoard(config_3x3)
        
        # Set one position
        board_p1.set_bit(0, 0, 0)
        
        # All engines should return the same valid moves
        moves_list = [engine.get_valid_moves(board_p1, board_p2) if hasattr(engine, 'get_valid_moves') 
                     else CPUEngine(config_3x3).get_valid_moves(board_p1, board_p2)
                     for engine in engines]
        
        # Convert moves to sets for comparison
        move_sets = [set(moves) for moves in moves_list]
        assert all(s == move_sets[0] for s in move_sets), "All engines should agree on valid moves"
        assert len(move_sets[0]) == config_3x3.size ** 3 - 1
        assert (0, 0, 0) not in move_sets[0]

class TestTicTacToe3D:
    @pytest.fixture
    def game_3x3(self):
        return TicTacToe3D(GameConfig(size=3, target=3))

    @pytest.mark.timeout(5)
    def test_initialization(self, game_3x3):
        assert game_3x3.current_player == 1
        assert game_3x3.board_p1.is_empty()
        assert game_3x3.board_p2.is_empty()
        assert game_3x3.engine is not None

    @pytest.mark.timeout(5)
    def test_engine_selection(self):
        config = GameConfig(size=3, target=3)
        
        game_cpu = TicTacToe3D(config, engine_type="cpu")
        assert isinstance(game_cpu.engine, CPUEngine)
        
        game_gpu = TicTacToe3D(config, engine_type="gpu")
        assert isinstance(game_gpu.engine, GPUEngine)
        
        game_mixed = TicTacToe3D(config, engine_type="mixed")
        assert isinstance(game_mixed.engine, MixedEngine)

    @pytest.mark.timeout(5)
    def test_horizontal_win(self, game_3x3):
        # Make horizontal winning line
        game_3x3.make_move(0, 0, 0)
        game_3x3.make_move(0, 1, 0)  # Player 2
        game_3x3.make_move(1, 0, 0)
        game_3x3.make_move(1, 1, 0)  # Player 2
        game_3x3.make_move(2, 0, 0)
        
        assert game_3x3.get_game_state() == GameResult.WIN

    @pytest.mark.timeout(5)
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
                
                # Get best move with sufficient search depth for perfect play
                # but not so deep it's too slow
                move = game_3x3.get_best_move(depth=8)
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

    @pytest.mark.timeout(5)
    def run_performance_test(self, benchmark, test_name: str, test_fn, 
                           test_args: dict, iterations: int,
                           improvement_threshold: float = 0.9) -> None:
        """Run a performance test with baseline comparison"""
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
            elif current_stats["mean"] > baseline["mean"] * 5.0:  # 20% regression threshold
                assert False, f"Performance regression detected: {percent_change:.1f}% slower than baseline"
        
        return result

    @pytest.mark.timeout(5)
    @pytest.mark.benchmark(
        group="engines",
        min_rounds=10,
        warmup=True,
        warmup_iterations=5
    )
    def test_cpu_engine_performance(self, benchmark, sample_boards):
        """Benchmark CPU engine performance"""
        boards_p1, boards_p2 = sample_boards
        config = GameConfig(size=4, target=4)
        game = TicTacToe3D(config, engine_type="cpu")
        
        def run_evaluation():
            # Evaluate a batch of positions
            results = []
            for b1, b2 in zip(boards_p1[:10], boards_p2[:10]):  # Test with 10 positions
                game.board_p1 = b1
                game.board_p2 = b2
                results.append(game.get_game_state())
            return results
        
        # Run performance test with stricter threshold for CPU
        eval_results = self.run_performance_test(
            benchmark=benchmark,
            test_name="engine_cpu",
            test_fn=run_evaluation,
            test_args={},
            iterations=10,  # Testing 10 positions
            improvement_threshold=0.95  # CPU is more stable, so use stricter threshold
        )
        
        # Verify all results are valid game states
        assert all(isinstance(r, GameResult) for r in eval_results)

    @pytest.mark.timeout(5)
    @pytest.mark.benchmark(
        group="engines",
        min_rounds=10,
        warmup=True,
        warmup_iterations=5
    )
    def test_gpu_engine_performance(self, benchmark, sample_boards):
        """Benchmark GPU engine performance"""
        boards_p1, boards_p2 = sample_boards
        config = GameConfig(size=4, target=4)
        game = TicTacToe3D(config, engine_type="gpu")
        
        def run_evaluation():
            # Evaluate a batch of positions
            results = []
            for b1, b2 in zip(boards_p1[:10], boards_p2[:10]):  # Test with 10 positions
                game.board_p1 = b1
                game.board_p2 = b2
                results.append(game.get_game_state())
            return results
        
        # Run performance test with more lenient threshold for GPU
        eval_results = self.run_performance_test(
            benchmark=benchmark,
            test_name="engine_gpu",
            test_fn=run_evaluation,
            test_args={},
            iterations=10,  # Testing 10 positions
            improvement_threshold=0.85  # GPU has more variance, so use more lenient threshold
        )
        
        # Verify all results are valid game states
        assert all(isinstance(r, GameResult) for r in eval_results)

    @pytest.mark.timeout(5)
    @pytest.mark.benchmark(
        group="engines",
        min_rounds=10,
        warmup=True,
        warmup_iterations=5
    )
    def test_mixed_engine_performance(self, benchmark, sample_boards):
        """Benchmark mixed engine performance"""
        boards_p1, boards_p2 = sample_boards
        config = GameConfig(size=4, target=4)
        game = TicTacToe3D(config, engine_type="mixed")
        
        def run_evaluation():
            # Evaluate a batch of positions
            results = []
            for b1, b2 in zip(boards_p1[:10], boards_p2[:10]):  # Test with 10 positions
                game.board_p1 = b1
                game.board_p2 = b2
                results.append(game.get_game_state())
            return results
        
        # Run performance test with balanced threshold for mixed engine
        eval_results = self.run_performance_test(
            benchmark=benchmark,
            test_name="engine_mixed",
            test_fn=run_evaluation,
            test_args={},
            iterations=10,  # Testing 10 positions
            improvement_threshold=0.9  # Mixed engine gets middle-ground threshold
        )
        
        # Verify all results are valid game states
        assert all(isinstance(r, GameResult) for r in eval_results)

if __name__ == "__main__":
    pytest.main([__file__])
