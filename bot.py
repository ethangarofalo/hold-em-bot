"""
Poker bot decision engine.
Uses Monte Carlo equity estimation, pot odds, position,
bet sizing, opponent modeling, and CFR-informed strategy
to make autonomous decisions.

v2: Now incorporates game-theoretic concepts from modern poker AI:
- Opponent modeling (exploit specific player tendencies)
- Information set abstraction (bucket similar situations)
- CFR-trained baseline strategy (regret-minimized play)
- Improved bet sizing (geometric sizing for stack commitment)
"""

from enum import Enum
from typing import List, Optional, Dict
from cards import Card, Rank
from monte_carlo import estimate_equity
from opponent_model import OpponentModel, OpponentStats
from info_sets import (
    equity_to_bucket, classify_board_texture, classify_spr,
    compute_preflop_strength, make_info_set_key, PositionBucket,
    SPRBucket, BoardTexture,
)
from cfr import CFRStrategy


class Action(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"


class Position(Enum):
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"      # cutoff, button
    BLINDS = "blinds"  # SB/BB


# Preflop hand categories (Chen formula simplified)
PREMIUM_HANDS = {
    ("A", "A"), ("K", "K"), ("Q", "Q"), ("J", "J"),
    ("A", "K"),  # suited or unsuited
}

STRONG_HANDS = {
    ("T", "T"), ("9", "9"), ("8", "8"),
    ("A", "Q"), ("A", "J"), ("K", "Q"),
}

PLAYABLE_HANDS = {
    ("7", "7"), ("6", "6"), ("5", "5"), ("4", "4"), ("3", "3"), ("2", "2"),
    ("A", "T"), ("K", "J"), ("Q", "J"), ("J", "T"),
    ("K", "T"), ("Q", "T"), ("T", "9"),
}


def _hand_category(hole_cards: List[Card]) -> str:
    """Categorize a preflop hand."""
    ranks = tuple(sorted(
        ["A" if c.rank == Rank.ACE else
         "K" if c.rank == Rank.KING else
         "Q" if c.rank == Rank.QUEEN else
         "J" if c.rank == Rank.JACK else
         "T" if c.rank == Rank.TEN else
         str(c.rank.value) for c in hole_cards],
        reverse=True
    ))
    suited = hole_cards[0].suit == hole_cards[1].suit

    if ranks in PREMIUM_HANDS or (ranks[1], ranks[0]) in PREMIUM_HANDS:
        return "premium"
    if ranks in STRONG_HANDS or (ranks[1], ranks[0]) in STRONG_HANDS:
        return "strong"
    if ranks in PLAYABLE_HANDS or (ranks[1], ranks[0]) in PLAYABLE_HANDS:
        if suited:
            return "playable_suited"
        return "playable"
    if suited and max(c.rank for c in hole_cards) >= Rank.TEN:
        return "speculative_suited"
    return "trash"


class PokerBot:
    """
    Autonomous No-Limit Texas Hold'em bot.

    Decision hierarchy:
    1. Check CFR strategy (game-theoretically sound baseline)
    2. Apply opponent-specific exploitative adjustments
    3. Fall back to heuristic strategy if no CFR data

    This mirrors how professional poker AIs work:
    a "blueprint" strategy (CFR) provides the GTO baseline,
    then real-time adjustments exploit specific opponents.
    """

    def __init__(self, name: str = "Bot", style: str = "balanced",
                 opponent_model: Optional[OpponentModel] = None,
                 cfr_strategy: Optional[CFRStrategy] = None):
        self.name = name
        self.style = style  # "tight", "balanced", "aggressive"
        self.opponent_model = opponent_model or OpponentModel()
        self.cfr_strategy = cfr_strategy

        # Style adjustments
        if style == "tight":
            self.aggression = 0.8
            self.bluff_frequency = 0.05
            self.open_threshold = 0.55
        elif style == "aggressive":
            self.aggression = 1.4
            self.bluff_frequency = 0.20
            self.open_threshold = 0.40
        else:  # balanced
            self.aggression = 1.0
            self.bluff_frequency = 0.12
            self.open_threshold = 0.45

    def decide(
        self,
        hole_cards: List[Card],
        community_cards: List[Card],
        pot: int,
        to_call: int,
        stack: int,
        position: Position,
        num_opponents: int,
        street: str,  # "preflop", "flop", "turn", "river"
        raise_count: int = 0,
        opponent_name: Optional[str] = None,
    ) -> tuple:
        """
        Make an autonomous decision.

        Returns:
            (Action, amount) — the action and bet/raise amount
        """
        # Get exploitative adjustments if we know the opponent
        adjustments = {}
        if opponent_name:
            adjustments = self.opponent_model.get_exploitative_adjustments(opponent_name)

        # Try CFR-informed decision first
        if self.cfr_strategy and self.cfr_strategy.iterations > 0:
            cfr_action = self._cfr_decision(
                hole_cards, community_cards, pot, to_call, stack,
                position, num_opponents, street, raise_count, adjustments
            )
            if cfr_action is not None:
                return cfr_action

        # Fall back to heuristic strategy
        if street == "preflop":
            return self._preflop_decision(
                hole_cards, pot, to_call, stack, position,
                num_opponents, raise_count, adjustments
            )
        else:
            return self._postflop_decision(
                hole_cards, community_cards, pot, to_call,
                stack, position, num_opponents, street,
                raise_count, adjustments
            )

    def _cfr_decision(
        self, hole_cards, community_cards, pot, to_call, stack,
        position, num_opponents, street, raise_count, adjustments
    ) -> Optional[tuple]:
        """
        Query the CFR strategy for a decision.
        Returns None if no strategy exists for this info set.
        """
        import random

        # Compute equity for bucketing
        if street == "preflop":
            equity = compute_preflop_strength(hole_cards)
        else:
            sims = 200  # fast estimate for lookup
            result = estimate_equity(
                hole_cards, community_cards,
                num_opponents=num_opponents, simulations=sims
            )
            equity = result["equity"]

        hand_bucket = equity_to_bucket(equity)
        board_texture = classify_board_texture(community_cards)
        spr_bucket = classify_spr(stack, pot)

        # Map position
        pos_map = {
            Position.EARLY: PositionBucket.EARLY,
            Position.MIDDLE: PositionBucket.MIDDLE,
            Position.LATE: PositionBucket.LATE,
            Position.BLINDS: PositionBucket.BLINDS,
        }
        pos_bucket = pos_map.get(position, PositionBucket.MIDDLE)

        info_key = make_info_set_key(
            hand_bucket, board_texture, spr_bucket,
            street, pos_bucket, to_call > 0, num_opponents
        )

        # Check if we have a trained strategy for this info set
        if info_key not in self.cfr_strategy.cumulative_strategy:
            return None

        actions = list(self.cfr_strategy.cumulative_strategy[info_key].keys())
        if not actions:
            return None

        avg_strategy = self.cfr_strategy.get_average_strategy(info_key, actions)

        # Apply exploitative adjustments to the CFR baseline
        adjusted_strategy = self._apply_adjustments(avg_strategy, adjustments)

        # Sample an action
        r = random.random()
        cumulative = 0.0
        chosen_action = actions[-1]
        for action in actions:
            cumulative += adjusted_strategy.get(action, 0)
            if r <= cumulative:
                chosen_action = action
                break

        # Convert CFR action to game action
        return self._cfr_action_to_game_action(
            chosen_action, pot, to_call, stack, equity
        )

    def _apply_adjustments(self, strategy: Dict[str, float],
                           adjustments: Dict[str, float]) -> Dict[str, float]:
        """Apply exploitative adjustments to a CFR strategy."""
        if not adjustments:
            return strategy

        adjusted = dict(strategy)
        bluff_mult = adjustments.get("bluff_more", 1.0)
        fold_mult = adjustments.get("fold_more", 1.0)
        raise_mult = adjustments.get("raise_more", 1.0)

        # Adjust raise probabilities (bluffing + value)
        for action in ["raise_half", "raise_pot", "all_in"]:
            if action in adjusted:
                adjusted[action] *= raise_mult * bluff_mult

        # Adjust fold probability
        if "fold" in adjusted:
            adjusted["fold"] *= fold_mult

        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {a: p / total for a, p in adjusted.items()}

        return adjusted

    def _cfr_action_to_game_action(
        self, cfr_action: str, pot: int, to_call: int,
        stack: int, equity: float
    ) -> tuple:
        """Convert a CFR abstract action to a concrete game action."""
        if cfr_action == "fold":
            return (Action.FOLD, 0)
        elif cfr_action == "check":
            return (Action.CHECK, 0)
        elif cfr_action == "call":
            return (Action.CALL, to_call)
        elif cfr_action == "raise_half":
            amount = max(int(pot * 0.5), to_call + 1)
            return (Action.RAISE, min(amount, stack))
        elif cfr_action == "raise_pot":
            amount = max(pot, to_call + 1)
            return (Action.RAISE, min(amount, stack))
        elif cfr_action == "all_in":
            return (Action.ALL_IN, stack)
        else:
            return (Action.CHECK, 0) if to_call == 0 else (Action.FOLD, 0)

    def _preflop_decision(
        self, hole_cards, pot, to_call, stack, position,
        num_opponents, raise_count, adjustments=None
    ) -> tuple:
        """Preflop strategy based on hand category and position."""
        adjustments = adjustments or {}
        category = _hand_category(hole_cards)
        big_blind = max(to_call, 1)  # Prevent division by zero

        # Get opponent-specific modifiers
        bluff_mult = adjustments.get("bluff_more", 1.0)
        raise_mult = adjustments.get("raise_more", 1.0)
        calling_adj = adjustments.get("tighten_calling", 0.0)

        # Premium hands: always raise/re-raise
        if category == "premium":
            if raise_count >= 2:
                if stack <= pot * 3:
                    return (Action.ALL_IN, stack)
                raise_size = min(int(pot * 3 * raise_mult), stack)
                return (Action.RAISE, raise_size)
            raise_size = min(max(pot, big_blind * 3), stack)
            return (Action.RAISE, int(raise_size * raise_mult))

        # Strong hands: raise from late, call from early
        if category == "strong":
            if raise_count >= 2:
                call_threshold = 0.15 - calling_adj
                if to_call <= stack * call_threshold:
                    return (Action.CALL, to_call)
                return (Action.FOLD, 0)
            if position in (Position.LATE, Position.MIDDLE):
                raise_size = min(max(big_blind * 3, pot), stack)
                return (Action.RAISE, int(raise_size * raise_mult))
            if to_call <= big_blind * 3:
                return (Action.CALL, to_call)
            return (Action.FOLD, 0)

        # Playable hands: position-dependent
        if category in ("playable", "playable_suited"):
            if raise_count >= 1 and position in (Position.EARLY, Position.BLINDS):
                return (Action.FOLD, 0)
            if position == Position.LATE and raise_count == 0:
                raise_size = min(big_blind * 3, stack)
                return (Action.RAISE, int(raise_size * raise_mult))
            max_call = big_blind * (2 - calling_adj * 10)
            if to_call <= max_call:
                return (Action.CALL, to_call)
            if category == "playable_suited" and to_call <= big_blind * 3:
                return (Action.CALL, to_call)
            return (Action.FOLD, 0)

        # Speculative suited: only from late position, cheap
        if category == "speculative_suited":
            max_call = big_blind * (2 - calling_adj * 5)
            if position == Position.LATE and to_call <= max_call:
                return (Action.CALL, to_call)
            return (Action.FOLD, 0)

        # Trash: fold (occasional bluff from late position)
        import random
        effective_bluff_freq = self.bluff_frequency * bluff_mult
        if (position == Position.LATE and raise_count == 0
                and random.random() < effective_bluff_freq):
            raise_size = min(big_blind * 3, stack)
            return (Action.RAISE, raise_size)

        if to_call == 0:
            return (Action.CHECK, 0)
        return (Action.FOLD, 0)

    def _postflop_decision(
        self, hole_cards, community_cards, pot, to_call,
        stack, position, num_opponents, street, raise_count,
        adjustments=None
    ) -> tuple:
        """
        Post-flop strategy driven by Monte Carlo equity vs pot odds.
        v2: incorporates board texture, SPR, and opponent adjustments.
        """
        import random
        adjustments = adjustments or {}

        # Run Monte Carlo simulation
        sims = 600 if street == "flop" else 400
        result = estimate_equity(
            hole_cards, community_cards,
            num_opponents=num_opponents,
            simulations=sims
        )
        equity = result["equity"]

        # Board texture analysis (new in v2)
        board_texture = classify_board_texture(community_cards)
        spr = classify_spr(stack, pot)

        # Opponent-specific adjustments
        bluff_mult = adjustments.get("bluff_more", 1.0)
        value_mult = adjustments.get("value_bet_more", 1.0)
        calling_adj = adjustments.get("tighten_calling", 0.0)
        fold_mult = adjustments.get("fold_more", 1.0)

        # Calculate pot odds
        if to_call > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0

        # Adjusted equity threshold (exploit-aware)
        equity_threshold_adj = calling_adj

        # === BET SIZING ENGINE (v2: geometric sizing) ===
        # On wet boards, size up. On dry boards, size down.
        texture_multiplier = {
            BoardTexture.DRY: 0.4,
            BoardTexture.MODERATE: 0.55,
            BoardTexture.WET: 0.7,
            BoardTexture.VERY_WET: 0.85,
            BoardTexture.PAIRED: 0.5,
        }.get(board_texture, 0.6)

        # SPR-aware commitment: shallow SPR = more willing to shove
        commit_threshold = {
            SPRBucket.SHALLOW: 0.45,    # pot-committed territory
            SPRBucket.MEDIUM: 0.55,
            SPRBucket.DEEP: 0.65,
        }.get(spr, 0.55)

        # === DECISION LOGIC ===

        # Monster hand (equity > 75%): bet/raise for value
        if equity > 0.75:
            if to_call > 0:
                if raise_count < 3 and stack > to_call * 2:
                    raise_amt = min(
                        int(pot * texture_multiplier * self.aggression * value_mult),
                        stack
                    )
                    return (Action.RAISE, max(raise_amt, to_call * 2))
                return (Action.CALL, to_call)
            # Leading out with value bet
            bet_size = int(pot * texture_multiplier * self.aggression * value_mult)
            bet_size = min(max(bet_size, 1), stack)
            return (Action.RAISE, bet_size)

        # Strong hand (55-75%): context-dependent
        if equity > 0.55 - equity_threshold_adj:
            if to_call > 0:
                if equity > pot_odds + 0.10 - equity_threshold_adj:
                    if (random.random() < 0.3 * self.aggression * value_mult
                            and raise_count < 2 and stack > to_call * 3):
                        raise_amt = min(int(pot * texture_multiplier), stack)
                        return (Action.RAISE, max(raise_amt, to_call * 2))
                    return (Action.CALL, to_call)
                if equity > pot_odds - equity_threshold_adj:
                    return (Action.CALL, to_call)
                return (Action.FOLD, 0)
            # Lead out
            bet_size = int(pot * texture_multiplier * 0.8 * self.aggression)
            bet_size = min(max(bet_size, 1), stack)
            return (Action.RAISE, bet_size)

        # Drawing / marginal (35-55%): call if odds justify
        if equity > 0.35 - equity_threshold_adj:
            if to_call > 0:
                if equity > pot_odds - equity_threshold_adj:
                    return (Action.CALL, to_call)
                # Semi-bluff raise on wet boards
                effective_bluff = self.bluff_frequency * self.aggression * bluff_mult
                if (random.random() < effective_bluff
                        and raise_count == 0 and street != "river"
                        and board_texture in (BoardTexture.WET, BoardTexture.VERY_WET)):
                    raise_amt = min(int(pot * texture_multiplier), stack)
                    return (Action.RAISE, max(raise_amt, to_call * 2))
                return (Action.FOLD, 0)
            # Check or probe bet
            probe_freq = 0.3 * self.aggression * bluff_mult
            if random.random() < probe_freq:
                bet_size = int(pot * texture_multiplier * 0.5)
                bet_size = min(max(bet_size, 1), stack)
                return (Action.RAISE, bet_size)
            return (Action.CHECK, 0)

        # Weak hand (equity < 35%)
        if to_call > 0:
            # River bluff (opponent-aware frequency)
            effective_bluff = self.bluff_frequency * bluff_mult
            if (random.random() < effective_bluff
                    and raise_count == 0 and street == "river"):
                raise_amt = min(int(pot * 0.7), stack)
                return (Action.RAISE, raise_amt)

            # Cheap call with implied odds at shallow SPR
            if (equity > pot_odds - equity_threshold_adj
                    and to_call <= stack * 0.05
                    and spr == SPRBucket.SHALLOW):
                return (Action.CALL, to_call)

            return (Action.FOLD, 0)

        # No bet to us — check (with occasional bluff)
        effective_bluff = self.bluff_frequency * bluff_mult
        if random.random() < effective_bluff:
            # Bluff more on dry boards (less likely opponent has a hand)
            if board_texture in (BoardTexture.DRY, BoardTexture.PAIRED):
                bet_size = min(int(pot * 0.5), stack)
                if bet_size > 0:
                    return (Action.RAISE, bet_size)
        return (Action.CHECK, 0)

    def __repr__(self):
        return f"PokerBot({self.name}, style={self.style})"
