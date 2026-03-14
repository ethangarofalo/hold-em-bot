"""
Opponent modeling for Texas Hold'em.
Tracks betting patterns per opponent and infers tendencies:
VPIP, PFR, aggression factor, fold-to-bet frequency, and more.

This moves the bot from treating all opponents identically
to exploiting specific weaknesses — the practical edge that
separates competent play from winning play.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class OpponentStats:
    """
    Statistical profile of a single opponent.

    Tracks:
    - VPIP (Voluntarily Put $ In Pot): % of hands they play
    - PFR (Pre-Flop Raise): % of hands they raise preflop
    - AF (Aggression Factor): (bets + raises) / calls
    - Fold to Bet: % of time they fold when facing a bet
    - Showdown frequency: how often they reach showdown
    - C-Bet frequency: how often they continuation bet the flop
    """

    def __init__(self):
        # Raw counters
        self.hands_observed = 0
        self.vpip_count = 0       # voluntarily put money in
        self.pfr_count = 0        # raised preflop
        self.calls = 0
        self.bets_and_raises = 0
        self.folds_to_bet = 0
        self.times_facing_bet = 0
        self.showdowns = 0
        self.showdown_wins = 0
        self.cbets = 0            # continuation bets
        self.cbet_opportunities = 0
        self.three_bets = 0       # 3-bet (re-raise preflop)
        self.three_bet_opportunities = 0

        # Per-street aggression
        self.street_bets = defaultdict(int)   # {"preflop": N, "flop": N, ...}
        self.street_calls = defaultdict(int)
        self.street_folds = defaultdict(int)
        self.street_checks = defaultdict(int)

        # Showdown hand strengths (for calibrating their ranges)
        self.shown_hand_strengths: List[float] = []

    @property
    def vpip(self) -> float:
        """Voluntarily Put $ In Pot (0.0 - 1.0)."""
        if self.hands_observed < 3:
            return 0.5  # unknown — assume average
        return self.vpip_count / self.hands_observed

    @property
    def pfr(self) -> float:
        """Pre-Flop Raise rate (0.0 - 1.0)."""
        if self.hands_observed < 3:
            return 0.15  # unknown — assume average
        return self.pfr_count / self.hands_observed

    @property
    def aggression_factor(self) -> float:
        """(bets + raises) / calls. Higher = more aggressive."""
        if self.calls == 0:
            return 2.0 if self.bets_and_raises > 0 else 1.0
        return self.bets_and_raises / self.calls

    @property
    def fold_to_bet_rate(self) -> float:
        """How often they fold when facing a bet/raise."""
        if self.times_facing_bet < 5:
            return 0.4  # unknown — assume moderate
        return self.folds_to_bet / self.times_facing_bet

    @property
    def cbet_rate(self) -> float:
        """Continuation bet frequency."""
        if self.cbet_opportunities < 3:
            return 0.65  # unknown — assume average
        return self.cbets / self.cbet_opportunities

    @property
    def three_bet_rate(self) -> float:
        """3-bet frequency."""
        if self.three_bet_opportunities < 5:
            return 0.08  # unknown — assume average
        return self.three_bets / self.three_bet_opportunities

    @property
    def showdown_win_rate(self) -> float:
        """Win rate at showdown."""
        if self.showdowns < 3:
            return 0.5
        return self.showdown_wins / self.showdowns

    def classify(self) -> str:
        """
        Classify opponent into a player type.
        Standard poker archetypes:
        - TAG (Tight-Aggressive): low VPIP, high AF
        - LAG (Loose-Aggressive): high VPIP, high AF
        - Nit: very low VPIP, low AF
        - Calling Station: high VPIP, low AF
        - Maniac: very high VPIP, very high AF
        """
        if self.hands_observed < 10:
            return "unknown"

        v = self.vpip
        af = self.aggression_factor

        if v < 0.20:
            return "nit" if af < 2.0 else "tag"
        elif v < 0.35:
            return "tag" if af >= 1.5 else "passive_reg"
        elif v < 0.50:
            return "lag" if af >= 2.0 else "calling_station"
        else:
            return "maniac" if af >= 2.5 else "calling_station"

    def record_action(self, action: str, street: str, facing_bet: bool = False,
                      is_voluntary: bool = True):
        """Record a single action for this opponent."""
        if action == "fold":
            self.street_folds[street] += 1
            if facing_bet:
                self.folds_to_bet += 1
                self.times_facing_bet += 1
        elif action == "call":
            self.calls += 1
            self.street_calls[street] += 1
            if facing_bet:
                self.times_facing_bet += 1
        elif action in ("raise", "all_in"):
            self.bets_and_raises += 1
            self.street_bets[street] += 1
            if facing_bet:
                self.times_facing_bet += 1
        elif action == "check":
            self.street_checks[street] += 1

    def record_hand_start(self, voluntarily_entered: bool, raised_preflop: bool):
        """Record the start of a new hand."""
        self.hands_observed += 1
        if voluntarily_entered:
            self.vpip_count += 1
        if raised_preflop:
            self.pfr_count += 1

    def record_showdown(self, won: bool, hand_strength: Optional[float] = None):
        """Record a showdown result."""
        self.showdowns += 1
        if won:
            self.showdown_wins += 1
        if hand_strength is not None:
            self.shown_hand_strengths.append(hand_strength)

    def record_cbet(self, did_cbet: bool):
        """Record a continuation bet opportunity."""
        self.cbet_opportunities += 1
        if did_cbet:
            self.cbets += 1

    def record_three_bet(self, did_three_bet: bool):
        """Record a 3-bet opportunity."""
        self.three_bet_opportunities += 1
        if did_three_bet:
            self.three_bets += 1

    def summary(self) -> str:
        """Human-readable summary of opponent stats."""
        if self.hands_observed < 3:
            return "Insufficient data"
        player_type = self.classify()
        return (
            f"Type: {player_type.upper()} | "
            f"VPIP: {self.vpip:.0%} | PFR: {self.pfr:.0%} | "
            f"AF: {self.aggression_factor:.1f} | "
            f"Fold%: {self.fold_to_bet_rate:.0%} | "
            f"Hands: {self.hands_observed}"
        )


class OpponentModel:
    """
    Tracks stats for all opponents at the table.
    The bot queries this to adjust strategy per-opponent.
    """

    def __init__(self):
        self.opponents: Dict[str, OpponentStats] = {}

    def get_stats(self, name: str) -> OpponentStats:
        """Get or create stats for a player."""
        if name not in self.opponents:
            self.opponents[name] = OpponentStats()
        return self.opponents[name]

    def get_exploitative_adjustments(self, opponent_name: str) -> Dict[str, float]:
        """
        Return strategy adjustments based on opponent tendencies.

        Returns multipliers/adjustments:
        - bluff_more: multiplier for bluff frequency (>1 = bluff more)
        - value_bet_more: multiplier for value betting
        - tighten_calling: adjustment to calling threshold
        - raise_more: multiplier for raise frequency
        """
        stats = self.get_stats(opponent_name)
        player_type = stats.classify()

        adjustments = {
            "bluff_more": 1.0,
            "value_bet_more": 1.0,
            "tighten_calling": 0.0,
            "raise_more": 1.0,
            "fold_more": 1.0,
        }

        if player_type == "unknown":
            return adjustments

        # Exploit nits: they fold too much — bluff them relentlessly
        if player_type == "nit":
            adjustments["bluff_more"] = 2.5
            adjustments["value_bet_more"] = 0.7  # they fold strong hands
            adjustments["raise_more"] = 1.5

        # Exploit calling stations: never bluff, value bet thin
        elif player_type == "calling_station":
            adjustments["bluff_more"] = 0.2   # don't bluff someone who calls everything
            adjustments["value_bet_more"] = 1.8  # bet thinner for value
            adjustments["tighten_calling"] = -0.05  # can call wider vs passive

        # Exploit TAGs: 3-bet them more, fold to their 4-bets
        elif player_type == "tag":
            adjustments["raise_more"] = 1.3
            adjustments["fold_more"] = 1.2  # respect their big bets

        # Exploit LAGs: call wider, trap more
        elif player_type == "lag":
            adjustments["tighten_calling"] = -0.08  # call wider
            adjustments["raise_more"] = 0.8   # trap more, raise less
            adjustments["bluff_more"] = 0.6

        # Exploit maniacs: call everything, let them bluff off their stack
        elif player_type == "maniac":
            adjustments["tighten_calling"] = -0.15
            adjustments["bluff_more"] = 0.1
            adjustments["value_bet_more"] = 1.5

        # Exploit passive regs: steal more, fold to their raises
        elif player_type == "passive_reg":
            adjustments["bluff_more"] = 1.5
            adjustments["raise_more"] = 1.3
            adjustments["fold_more"] = 1.3  # when they raise, they have it

        # Fine-tune based on specific stats if we have enough data
        if stats.hands_observed >= 20:
            # High fold-to-bet → bluff more
            if stats.fold_to_bet_rate > 0.55:
                adjustments["bluff_more"] *= 1.3
            elif stats.fold_to_bet_rate < 0.25:
                adjustments["bluff_more"] *= 0.5

            # High VPIP but low showdown win rate → they play bad hands
            if stats.vpip > 0.45 and stats.showdown_win_rate < 0.40:
                adjustments["value_bet_more"] *= 1.3

        return adjustments

    def display_all(self) -> str:
        """Display all opponent profiles."""
        lines = ["  OPPONENT PROFILES"]
        lines.append("  " + "-" * 60)
        for name, stats in sorted(self.opponents.items()):
            lines.append(f"  {name:12s}: {stats.summary()}")
        return "\n".join(lines)
