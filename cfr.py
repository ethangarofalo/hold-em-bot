"""
Simplified Counterfactual Regret Minimization (CFR) for Hold'em.

CFR is the algorithm behind every modern poker AI — Libratus,
Pluribus, and fedden/poker_ai all use variants of it. The core
idea: play millions of hands against yourself, track how much
you "regret" not taking each action, then update your strategy
proportional to that regret. Over enough iterations, this
converges to a Nash equilibrium — the theoretically unexploitable
strategy.

This implementation uses:
- External sampling MCCFR (Monte Carlo CFR): instead of traversing
  the entire game tree, we sample random deals and only update
  the branches we actually visit. Much faster convergence for
  large games.
- Information set abstraction (from info_sets.py): groups similar
  game states together so the strategy space is tractable.
- Regret matching: the standard technique for converting cumulative
  regrets into a probability distribution over actions.

The trained strategy is stored as a dictionary mapping
information set keys to action probabilities. The bot can
then look up its current situation and play accordingly.

Zero external dependencies. Pure Python.
"""

import random
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from cards import Card, Deck, Rank, Suit, evaluate_hand
from info_sets import (
    HandBucket, BoardTexture, SPRBucket, PositionBucket,
    equity_to_bucket, classify_board_texture, classify_spr,
    compute_preflop_strength, make_info_set_key, get_available_actions,
)
from monte_carlo import estimate_equity


class CFRStrategy:
    """
    Stores and updates the CFR strategy.

    For each information set, tracks:
    - cumulative_regrets: how much we regret not playing each action
    - cumulative_strategy: sum of all strategies used (for averaging)
    - current_strategy: the live strategy (probability per action)
    """

    def __init__(self):
        # info_set_key -> {action: cumulative_regret}
        self.cumulative_regrets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # info_set_key -> {action: cumulative_strategy_weight}
        self.cumulative_strategy: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.iterations = 0

    def get_strategy(self, info_set_key: str, actions: List[str]) -> Dict[str, float]:
        """
        Compute current strategy via regret matching.

        The probability of playing action 'a' is proportional to
        max(0, cumulative_regret[a]). If all regrets are negative,
        play uniformly.
        """
        regrets = self.cumulative_regrets[info_set_key]

        # Positive regrets only
        positive_regrets = {a: max(0.0, regrets[a]) for a in actions}
        total = sum(positive_regrets.values())

        if total > 0:
            strategy = {a: positive_regrets[a] / total for a in actions}
        else:
            # Uniform random when no positive regrets
            n = len(actions)
            strategy = {a: 1.0 / n for a in actions}

        return strategy

    def get_average_strategy(self, info_set_key: str, actions: List[str]) -> Dict[str, float]:
        """
        Get the average strategy over all iterations.
        This is what converges to Nash equilibrium —
        NOT the current strategy, but the cumulative average.
        """
        cum = self.cumulative_strategy[info_set_key]
        total = sum(cum[a] for a in actions)

        if total > 0:
            return {a: cum[a] / total for a in actions}
        else:
            n = len(actions)
            return {a: 1.0 / n for a in actions}

    def update_regrets(self, info_set_key: str, action_values: Dict[str, float],
                       strategy: Dict[str, float]):
        """
        Update cumulative regrets for an information set.

        For each action, regret = value(action) - value(strategy).
        The value of the strategy is the weighted sum of action values.
        """
        strategy_value = sum(
            strategy[a] * action_values.get(a, 0.0)
            for a in strategy
        )

        for action in strategy:
            regret = action_values.get(action, 0.0) - strategy_value
            self.cumulative_regrets[info_set_key][action] += regret

    def update_cumulative_strategy(self, info_set_key: str,
                                    strategy: Dict[str, float],
                                    reach_probability: float = 1.0):
        """Accumulate the strategy weighted by reach probability."""
        for action, prob in strategy.items():
            self.cumulative_strategy[info_set_key][action] += reach_probability * prob

    def sample_action(self, info_set_key: str, actions: List[str]) -> str:
        """Sample an action according to the current strategy."""
        strategy = self.get_strategy(info_set_key, actions)
        r = random.random()
        cumulative = 0.0
        for action in actions:
            cumulative += strategy[action]
            if r <= cumulative:
                return action
        return actions[-1]

    def save(self, filepath: str):
        """Save the trained strategy to a JSON file."""
        data = {
            "iterations": self.iterations,
            "strategy": {},
        }
        for key in self.cumulative_strategy:
            actions = list(self.cumulative_strategy[key].keys())
            if actions:
                avg = self.get_average_strategy(key, actions)
                data["strategy"][key] = avg

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load a pre-trained strategy from JSON."""
        if not os.path.exists(filepath):
            return False

        with open(filepath, "r") as f:
            data = json.load(f)

        self.iterations = data.get("iterations", 0)

        # Load into cumulative_strategy so get_average_strategy works
        for key, action_probs in data.get("strategy", {}).items():
            for action, prob in action_probs.items():
                self.cumulative_strategy[key][action] = prob * 1000  # scale up

        return True

    def stats(self) -> str:
        """Summary of trained strategy."""
        n_sets = len(self.cumulative_strategy)
        n_regrets = len(self.cumulative_regrets)
        return (
            f"CFR Strategy: {self.iterations} iterations, "
            f"{n_sets} info sets trained, "
            f"{n_regrets} regret entries"
        )


class CFRTrainer:
    """
    Trains a CFR strategy through self-play.

    Uses external sampling MCCFR: for each iteration,
    deal random cards and play through the hand, updating
    regrets at each decision point.
    """

    def __init__(self, strategy: Optional[CFRStrategy] = None):
        self.strategy = strategy or CFRStrategy()

    def train(self, iterations: int = 10000, verbose: bool = True):
        """
        Run MCCFR training iterations.

        Each iteration:
        1. Deal random hole cards to two players
        2. Deal random community cards
        3. At each decision point, compute the information set
        4. Play according to current strategy
        5. At showdown, compute payoffs
        6. Walk back through the hand, updating regrets
        """
        if verbose:
            print(f"  Starting CFR training ({iterations} iterations)...")

        for i in range(iterations):
            self.strategy.iterations += 1

            # Deal random cards
            deck_cards = [Card(r, s) for r in Rank for s in Suit]
            random.shuffle(deck_cards)

            p1_hole = [deck_cards[0], deck_cards[1]]
            p2_hole = [deck_cards[2], deck_cards[3]]
            community = deck_cards[4:9]  # flop + turn + river

            # Train from player 1's perspective
            self._train_hand(p1_hole, p2_hole, community)

            # Train from player 2's perspective (symmetry)
            self._train_hand(p2_hole, p1_hole, community)

            if verbose and (i + 1) % max(1, iterations // 10) == 0:
                pct = (i + 1) / iterations * 100
                print(f"    {pct:.0f}% complete ({i + 1}/{iterations})")

        if verbose:
            print(f"  Training complete. {self.strategy.stats()}")

    def _train_hand(self, hero_hole: List[Card], villain_hole: List[Card],
                    community: List[Card]):
        """
        Train a single hand from hero's perspective.
        Simplified: we simulate key decision points and update regrets.
        """
        pot = 15  # assume standard blinds
        stack = 1000

        streets = [
            ("preflop", []),
            ("flop", community[:3]),
            ("turn", community[:4]),
            ("river", community[:5]),
        ]

        for street_name, board in streets:
            # Compute hero's equity
            if street_name == "preflop":
                equity = compute_preflop_strength(hero_hole)
            else:
                eq_result = estimate_equity(
                    hero_hole, board, num_opponents=1, simulations=100
                )
                equity = eq_result["equity"]

            hand_bucket = equity_to_bucket(equity)
            board_texture = classify_board_texture(board)
            spr_bucket = classify_spr(stack, pot)

            # Create info set keys for both facing-bet and not-facing-bet
            for facing_bet in [False, True]:
                to_call = int(pot * 0.5) if facing_bet else 0
                actions = get_available_actions(to_call, stack, raise_count=0)

                if not actions:
                    continue

                info_key = make_info_set_key(
                    hand_bucket, board_texture, spr_bucket,
                    street_name, PositionBucket.LATE, facing_bet,
                    num_opponents=1,
                )

                strategy = self.strategy.get_strategy(info_key, actions)
                self.strategy.update_cumulative_strategy(info_key, strategy)

                # Compute action values based on equity
                action_values = self._compute_action_values(
                    equity, pot, to_call, stack, actions, facing_bet
                )

                self.strategy.update_regrets(info_key, action_values, strategy)

    def _compute_action_values(
        self, equity: float, pot: int, to_call: int, stack: int,
        actions: List[str], facing_bet: bool,
    ) -> Dict[str, float]:
        """
        Estimate the expected value of each action.
        Uses equity and pot geometry to approximate outcomes.
        """
        values = {}

        for action in actions:
            if action == "fold":
                values[action] = 0.0  # lose nothing more

            elif action == "check":
                # EV of checking ≈ equity * pot
                values[action] = equity * pot

            elif action == "call":
                # EV of calling = equity * (pot + to_call) - (1-equity) * to_call
                values[action] = equity * (pot + to_call) - (1 - equity) * to_call

            elif action == "raise_half":
                raise_to = int(pot * 0.5)
                # Simplified: assume opponent folds ~40% of the time
                fold_equity = 0.4
                # EV = fold_equity * pot + (1-fold_equity) * [equity * new_pot - cost]
                new_pot = pot + raise_to * 2
                ev_called = equity * new_pot - (1 - equity) * raise_to
                values[action] = fold_equity * pot + (1 - fold_equity) * ev_called

            elif action == "raise_pot":
                raise_to = pot
                fold_equity = 0.5
                new_pot = pot + raise_to * 2
                ev_called = equity * new_pot - (1 - equity) * raise_to
                values[action] = fold_equity * pot + (1 - fold_equity) * ev_called

            elif action == "all_in":
                fold_equity = 0.6  # bigger bets get more folds
                new_pot = pot + stack * 2
                ev_called = equity * new_pot - (1 - equity) * stack
                values[action] = fold_equity * pot + (1 - fold_equity) * ev_called

        return values


def train_and_save(iterations: int = 10000, filepath: str = "strategy.json"):
    """Convenience function to train and save a strategy."""
    trainer = CFRTrainer()
    trainer.train(iterations=iterations)
    trainer.strategy.save(filepath)
    print(f"\n  Strategy saved to {filepath}")
    return trainer.strategy


if __name__ == "__main__":
    print("=" * 50)
    print("  CFR TRAINING")
    print("  Monte Carlo Counterfactual Regret Minimization")
    print("=" * 50)
    print()

    strategy = train_and_save(iterations=5000)

    # Show some example strategies
    print("\n  SAMPLE LEARNED STRATEGIES:")
    print("  " + "-" * 40)
    for key in sorted(list(strategy.cumulative_strategy.keys()))[:10]:
        actions = list(strategy.cumulative_strategy[key].keys())
        avg = strategy.get_average_strategy(key, actions)
        print(f"  {key}:")
        for action, prob in sorted(avg.items(), key=lambda x: -x[1]):
            if prob > 0.01:
                print(f"    {action:12s}: {prob:.1%}")
