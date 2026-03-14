"""
Card, Deck, and Hand representation for Texas Hold'em.
Clean, fast, and built for Monte Carlo simulation.
"""

import random
from enum import IntEnum
from typing import List, Tuple, Optional
from itertools import combinations


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


SUIT_SYMBOLS = {Suit.CLUBS: "♣", Suit.DIAMONDS: "♦", Suit.HEARTS: "♥", Suit.SPADES: "♠"}
RANK_SYMBOLS = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A"
}

RANK_FROM_CHAR = {v: k for k, v in RANK_SYMBOLS.items()}
SUIT_FROM_CHAR = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}


class Card:
    """A single playing card."""

    __slots__ = ('rank', 'suit')

    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    @classmethod
    def from_str(cls, s: str) -> 'Card':
        """Parse a card from string like 'Ah', 'Td', '2c'."""
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s}")
        rank = RANK_FROM_CHAR.get(s[0].upper())
        suit = SUIT_FROM_CHAR.get(s[1].lower())
        if rank is None or suit is None:
            raise ValueError(f"Invalid card string: {s}")
        return cls(rank, suit)

    def __repr__(self):
        return f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __lt__(self, other):
        return (self.rank, self.suit) < (other.rank, other.suit)


class Deck:
    """A standard 52-card deck with efficient dealing."""

    def __init__(self):
        self.cards = [Card(r, s) for r in Rank for s in Suit]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> List[Card]:
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self.cards)} remain")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def remove(self, cards: List[Card]):
        """Remove specific cards from the deck (for Monte Carlo)."""
        card_set = set((c.rank, c.suit) for c in cards)
        self.cards = [c for c in self.cards if (c.rank, c.suit) not in card_set]

    def __len__(self):
        return len(self.cards)


class HandRank(IntEnum):
    """Hand rankings from worst to best."""
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


HAND_NAMES = {
    HandRank.HIGH_CARD: "High Card",
    HandRank.ONE_PAIR: "One Pair",
    HandRank.TWO_PAIR: "Two Pair",
    HandRank.THREE_OF_A_KIND: "Three of a Kind",
    HandRank.STRAIGHT: "Straight",
    HandRank.FLUSH: "Flush",
    HandRank.FULL_HOUSE: "Full House",
    HandRank.FOUR_OF_A_KIND: "Four of a Kind",
    HandRank.STRAIGHT_FLUSH: "Straight Flush",
    HandRank.ROYAL_FLUSH: "Royal Flush",
}


def evaluate_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
    """
    Evaluate the best 5-card hand from any number of cards (5-7).
    Returns (HandRank, kickers) where kickers is a list of ranks
    for tiebreaking, ordered from most significant to least.
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")

    best = None
    for combo in combinations(cards, 5):
        result = _evaluate_five(list(combo))
        if best is None or result > best:
            best = result
    return best


def _evaluate_five(cards: List[Card]) -> Tuple[HandRank, List[int]]:
    """Evaluate exactly 5 cards."""
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1

    # Check for straight (including A-2-3-4-5 wheel)
    is_straight = False
    straight_high = 0
    unique_ranks = sorted(set(ranks), reverse=True)

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:  # Ace-low straight (wheel)
            is_straight = True
            straight_high = 5

    # Count rank frequencies
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    # Sort by (count, rank) descending for kicker ordering
    sorted_ranks = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = [c for _, c in sorted_ranks]
    kicker_ranks = [r for r, _ in sorted_ranks]

    # Determine hand rank
    if is_straight and is_flush:
        if straight_high == 14:
            return (HandRank.ROYAL_FLUSH, [14])
        return (HandRank.STRAIGHT_FLUSH, [straight_high])

    if counts == [4, 1]:
        return (HandRank.FOUR_OF_A_KIND, kicker_ranks)

    if counts == [3, 2]:
        return (HandRank.FULL_HOUSE, kicker_ranks)

    if is_flush:
        return (HandRank.FLUSH, ranks)

    if is_straight:
        return (HandRank.STRAIGHT, [straight_high])

    if counts == [3, 1, 1]:
        return (HandRank.THREE_OF_A_KIND, kicker_ranks)

    if counts == [2, 2, 1]:
        return (HandRank.TWO_PAIR, kicker_ranks)

    if counts == [2, 1, 1, 1]:
        return (HandRank.ONE_PAIR, kicker_ranks)

    return (HandRank.HIGH_CARD, ranks)


def hand_to_str(hand_rank: HandRank, kickers: List[int]) -> str:
    """Human-readable hand description."""
    return HAND_NAMES[hand_rank]
