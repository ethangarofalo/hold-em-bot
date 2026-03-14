"""
Information set abstraction for Texas Hold'em.

In game theory, an "information set" is everything a player knows
at a decision point. In poker, two situations that look the same
to the player (same hand strength bucket, similar board texture,
similar pot odds) should produce the same strategy.

This module buckets the infinite space of poker situations into
tractable categories that CFR can train over and the bot can
look up at decision time.

Inspired by the abstraction techniques in:
- fedden/poker_ai (MCCFR with card abstraction)
- Libratus (blueprint + real-time refinement)
- Pluribus (depth-limited search with abstraction)
"""

from typing import List, Tuple, Optional
from enum import IntEnum
from cards import Card, Rank, Suit, evaluate_hand, HandRank


class HandBucket(IntEnum):
    """
    Coarse hand strength categories.
    Used pre-flop and refined post-flop with equity.
    """
    TRASH = 0        # bottom 20% — fold unless free
    WEAK = 1         # 20-35% — speculative / drawing
    MARGINAL = 2     # 35-50% — call-worthy
    GOOD = 3         # 50-65% — raise-worthy
    STRONG = 4       # 65-80% — strong value
    MONSTER = 5      # 80%+ — near nuts


class BoardTexture(IntEnum):
    """Board texture categories that affect strategy."""
    DRY = 0          # rainbow, disconnected (e.g., K-7-2 rainbow)
    MODERATE = 1     # some draws possible (one suit, one connected)
    WET = 2          # flush draw + straight draw possible
    VERY_WET = 3     # monotone or highly connected
    PAIRED = 4       # paired board changes dynamics


class PositionBucket(IntEnum):
    """Simplified position categories."""
    EARLY = 0
    MIDDLE = 1
    LATE = 2
    BLINDS = 3


class SPRBucket(IntEnum):
    """Stack-to-pot ratio categories."""
    SHALLOW = 0      # SPR < 3 — commit or fold territory
    MEDIUM = 1       # SPR 3-8 — standard play
    DEEP = 2         # SPR > 8 — implied odds matter


def equity_to_bucket(equity: float) -> HandBucket:
    """Convert raw equity to a hand strength bucket."""
    if equity >= 0.80:
        return HandBucket.MONSTER
    elif equity >= 0.65:
        return HandBucket.STRONG
    elif equity >= 0.50:
        return HandBucket.GOOD
    elif equity >= 0.35:
        return HandBucket.MARGINAL
    elif equity >= 0.20:
        return HandBucket.WEAK
    else:
        return HandBucket.TRASH


def classify_board_texture(community_cards: List[Card]) -> BoardTexture:
    """
    Classify the board texture from community cards.
    Considers flush draws, straight draws, and paired boards.
    """
    if len(community_cards) < 3:
        return BoardTexture.DRY

    suits = [c.suit for c in community_cards]
    ranks = sorted([c.rank for c in community_cards])

    # Check for paired board
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    if max(rank_counts.values()) >= 2:
        return BoardTexture.PAIRED

    # Flush draw potential
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suited = max(suit_counts.values())
    flush_draw = max_suited >= 3  # monotone or close

    # Straight draw potential: count connected cards
    unique_ranks = sorted(set(ranks))
    gaps = 0
    for i in range(len(unique_ranks) - 1):
        gap = unique_ranks[i + 1] - unique_ranks[i]
        if gap <= 2:
            gaps += 1

    connected = gaps >= 2

    # Classify
    if flush_draw and connected:
        return BoardTexture.VERY_WET
    elif flush_draw or connected:
        return BoardTexture.WET
    elif gaps >= 1:
        return BoardTexture.MODERATE
    else:
        return BoardTexture.DRY


def classify_spr(stack: int, pot: int) -> SPRBucket:
    """Classify stack-to-pot ratio."""
    if pot <= 0:
        return SPRBucket.DEEP
    spr = stack / pot
    if spr < 3:
        return SPRBucket.SHALLOW
    elif spr < 8:
        return SPRBucket.MEDIUM
    else:
        return SPRBucket.DEEP


def compute_preflop_strength(hole_cards: List[Card]) -> float:
    """
    Fast preflop hand strength heuristic (0.0 - 1.0).
    Based on a simplified Chen formula.
    No Monte Carlo needed — just card properties.
    """
    r1, r2 = sorted([c.rank for c in hole_cards], reverse=True)
    suited = hole_cards[0].suit == hole_cards[1].suit

    # Base score from high card
    score = r1.value  # 2-14

    # Pair bonus
    if r1 == r2:
        score = max(score * 2, 5)
        # High pairs are premium
        if r1 >= Rank.JACK:
            score += 4
        if r1 >= Rank.ACE:
            score += 2

    # Suited bonus
    if suited:
        score += 2

    # Gap penalty (connectivity)
    gap = r1.value - r2.value - 1
    if gap == 0:
        score += 1  # connected
    elif gap == 1:
        score += 0  # one-gapper
    elif gap <= 3:
        score -= gap
    else:
        score -= 4

    # High card kicker bonus
    if r2 >= Rank.QUEEN:
        score += 1
    if r1 >= Rank.ACE:
        score += 1

    # Normalize to 0.0 - 1.0 (raw range roughly 0-32)
    normalized = max(0.0, min(1.0, (score - 2) / 28))
    return normalized


def make_info_set_key(
    hand_bucket: HandBucket,
    board_texture: BoardTexture,
    spr_bucket: SPRBucket,
    street: str,
    position_bucket: PositionBucket,
    facing_bet: bool,
    num_opponents: int,
) -> str:
    """
    Create a hashable information set key.
    This is what CFR trains over — each unique key
    maps to a strategy (probability distribution over actions).
    """
    opp_bucket = min(num_opponents, 3)  # 1, 2, 3+
    facing = "F" if facing_bet else "N"
    return (
        f"{street[0].upper()}"       # P/F/T/R
        f"-H{hand_bucket.value}"     # hand strength 0-5
        f"-B{board_texture.value}"   # board texture 0-4
        f"-S{spr_bucket.value}"      # SPR 0-2
        f"-{position_bucket.name[0]}" # E/M/L/B
        f"-{facing}"                 # facing bet or not
        f"-O{opp_bucket}"            # opponent count
    )


def get_available_actions(to_call: int, stack: int, raise_count: int) -> List[str]:
    """
    Get the list of available actions at a decision point.
    Simplified action space for tractable CFR.
    """
    actions = []

    if to_call == 0:
        actions.append("check")
    else:
        actions.append("fold")
        if stack >= to_call:
            actions.append("call")

    # Raise options (simplified: half-pot, pot, all-in)
    if stack > to_call and raise_count < 4:
        actions.append("raise_half")
        actions.append("raise_pot")

    if stack > 0:
        actions.append("all_in")

    return actions
