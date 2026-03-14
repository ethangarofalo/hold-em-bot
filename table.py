"""
Poker table — runs a full No-Limit Texas Hold'em game
with multiple bot players.

v2: Integrates opponent modeling so bots track and exploit
each other's tendencies over the course of a session.
"""

import random
from typing import List, Optional, Dict
from cards import Card, Deck, evaluate_hand, hand_to_str, HAND_NAMES
from bot import PokerBot, Action, Position
from opponent_model import OpponentModel


class Player:
    """A player at the table."""

    def __init__(self, bot: PokerBot, stack: int):
        self.bot = bot
        self.name = bot.name
        self.stack = stack
        self.hole_cards: List[Card] = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False
        self.total_invested = 0  # total chips put in this hand
        self.voluntarily_entered = False  # for VPIP tracking
        self.raised_preflop = False       # for PFR tracking
        self.was_preflop_raiser = False    # for c-bet tracking

    def reset_hand(self):
        self.hole_cards = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False
        self.total_invested = 0
        self.voluntarily_entered = False
        self.raised_preflop = False
        self.was_preflop_raiser = False

    def is_active(self) -> bool:
        return not self.folded and not self.all_in and self.stack > 0

    def __repr__(self):
        status = ""
        if self.folded:
            status = " [FOLDED]"
        elif self.all_in:
            status = " [ALL-IN]"
        return f"{self.name}: ${self.stack}{status}"


class Table:
    """
    Manages a full No-Limit Texas Hold'em game.
    Handles dealing, betting rounds, pot management,
    showdown, and opponent modeling updates.
    """

    def __init__(
        self,
        players: List[Player],
        small_blind: int = 5,
        big_blind: int = 10,
        verbose: bool = True,
    ):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.verbose = verbose
        self.dealer_idx = 0
        self.community_cards: List[Card] = []
        self.pot = 0
        self.deck = Deck()
        self.hand_number = 0
        self.log: List[str] = []

        # Shared opponent model — all bots share the same tracker
        self.opponent_model = OpponentModel()
        for p in players:
            p.bot.opponent_model = self.opponent_model

    def _log(self, msg: str):
        self.log.append(msg)
        if self.verbose:
            print(msg)

    def _active_players(self) -> List[Player]:
        return [p for p in self.players if not p.folded]

    def _players_who_can_act(self) -> List[Player]:
        return [p for p in self.players if p.is_active()]

    def _get_position(self, player_idx: int) -> Position:
        """Determine position relative to the dealer."""
        n = len(self.players)
        relative = (player_idx - self.dealer_idx) % n
        if n <= 3:
            if relative == 0:
                return Position.LATE
            return Position.BLINDS
        if relative <= 1:
            return Position.BLINDS
        if relative <= n // 3:
            return Position.EARLY
        if relative <= 2 * n // 3:
            return Position.MIDDLE
        return Position.LATE

    def _get_primary_opponent(self, player: Player) -> Optional[str]:
        """Get the main opponent's name (for heads-up adjustments)."""
        active = [p for p in self.players if not p.folded and p is not player]
        if len(active) == 1:
            return active[0].name
        # In multiway, return the most active player
        if active:
            return active[0].name
        return None

    def _post_blinds(self) -> int:
        """Post small and big blinds. Returns index of player after BB."""
        n = len(self.players)
        sb_idx = (self.dealer_idx + 1) % n
        bb_idx = (self.dealer_idx + 2) % n

        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]

        sb_amount = min(self.small_blind, sb_player.stack)
        bb_amount = min(self.big_blind, bb_player.stack)

        sb_player.stack -= sb_amount
        sb_player.current_bet = sb_amount
        sb_player.total_invested += sb_amount

        bb_player.stack -= bb_amount
        bb_player.current_bet = bb_amount
        bb_player.total_invested += bb_amount

        self.pot = sb_amount + bb_amount

        self._log(f"  {sb_player.name} posts small blind: ${sb_amount}")
        self._log(f"  {bb_player.name} posts big blind: ${bb_amount}")

        return (bb_idx + 1) % n

    def _deal_hole_cards(self):
        """Deal 2 cards to each player."""
        for p in self.players:
            p.hole_cards = self.deck.deal(2)
            if self.verbose:
                self._log(f"  {p.name} dealt: {p.hole_cards[0]} {p.hole_cards[1]}")

    def _deal_community(self, n: int, label: str):
        """Deal community cards."""
        if n > 0:
            # Burn one card
            self.deck.deal(1)
        cards = self.deck.deal(n)
        self.community_cards.extend(cards)
        card_str = " ".join(str(c) for c in self.community_cards)
        self._log(f"\n  === {label} === [{card_str}]")

    def _betting_round(self, start_idx: int, street: str) -> bool:
        """
        Run a single betting round.
        Returns True if hand continues, False if only one player remains.
        """
        n = len(self.players)
        current_bet = max(p.current_bet for p in self.players)
        raise_count = 0
        last_raiser = None
        acted = set()
        preflop_raiser_name = None

        idx = start_idx
        while True:
            player = self.players[idx % n]

            # Skip folded, all-in, or eliminated players
            if player.folded or player.all_in or player.stack <= 0:
                idx += 1
                if idx % n == (start_idx if last_raiser is None else last_raiser) % n:
                    break
                if len(acted) >= n:
                    break
                continue

            to_call = current_bet - player.current_bet
            num_opponents = len([p for p in self.players if not p.folded and p is not player])
            position = self._get_position(idx % n)
            opponent_name = self._get_primary_opponent(player)

            action, amount = player.bot.decide(
                hole_cards=player.hole_cards,
                community_cards=self.community_cards,
                pot=self.pot,
                to_call=to_call,
                stack=player.stack,
                position=position,
                num_opponents=num_opponents,
                street=street,
                raise_count=raise_count,
                opponent_name=opponent_name,
            )

            # Record action for opponent modeling
            facing_bet = to_call > 0
            action_str = action.value

            for other in self.players:
                if other is not player and not other.folded:
                    stats = self.opponent_model.get_stats(player.name)
                    stats.record_action(action_str, street, facing_bet)

            # Execute the action
            if action == Action.FOLD:
                player.folded = True
                self._log(f"  {player.name} folds")

            elif action == Action.CHECK:
                self._log(f"  {player.name} checks")

            elif action == Action.CALL:
                call_amount = min(to_call, player.stack)
                player.stack -= call_amount
                player.current_bet += call_amount
                player.total_invested += call_amount
                self.pot += call_amount
                if street == "preflop":
                    player.voluntarily_entered = True
                if player.stack == 0:
                    player.all_in = True
                    self._log(f"  {player.name} calls ${call_amount} (ALL-IN)")
                else:
                    self._log(f"  {player.name} calls ${call_amount}")

            elif action in (Action.RAISE, Action.ALL_IN):
                if action == Action.ALL_IN:
                    amount = player.stack + player.current_bet

                # Ensure minimum raise
                min_raise = current_bet + self.big_blind
                actual_raise = max(amount, min_raise)
                actual_raise = min(actual_raise, player.stack + player.current_bet)

                chips_to_add = actual_raise - player.current_bet
                chips_to_add = min(chips_to_add, player.stack)

                player.stack -= chips_to_add
                player.current_bet += chips_to_add
                player.total_invested += chips_to_add
                self.pot += chips_to_add
                current_bet = player.current_bet

                if street == "preflop":
                    player.voluntarily_entered = True
                    player.raised_preflop = True
                    preflop_raiser_name = player.name

                    # Track 3-bet opportunities
                    if raise_count == 1:
                        for other in self.players:
                            if other is not player and not other.folded:
                                self.opponent_model.get_stats(player.name).record_three_bet(True)

                raise_count += 1
                last_raiser = idx

                if player.stack == 0:
                    player.all_in = True
                    self._log(f"  {player.name} raises to ${player.current_bet} (ALL-IN)")
                else:
                    self._log(f"  {player.name} raises to ${player.current_bet}")

                # Reset acted set — everyone needs to act again
                acted = {idx % n}
                idx += 1
                continue

            acted.add(idx % n)

            # Check if only one player remains
            active = [p for p in self.players if not p.folded]
            if len(active) <= 1:
                return False

            # Check if round is complete
            idx += 1
            all_matched = all(
                p.current_bet == current_bet or p.folded or p.all_in
                for p in self.players
            )
            all_acted = len(acted) >= len(self._players_who_can_act())

            if all_matched and all_acted:
                break

            # Safety: prevent infinite loops
            if len(acted) > n * 4:
                break

        # Track preflop raiser for c-bet detection
        if street == "preflop" and preflop_raiser_name:
            for p in self.players:
                if p.name == preflop_raiser_name:
                    p.was_preflop_raiser = True

        return True

    def _reset_bets(self):
        """Reset current bets for a new street."""
        for p in self.players:
            p.current_bet = 0

    def _showdown(self):
        """Evaluate hands and award pot."""
        active = [p for p in self.players if not p.folded]

        if len(active) == 1:
            winner = active[0]
            winner.stack += self.pot
            self._log(f"\n  {winner.name} wins ${self.pot} (everyone else folded)")
            return

        self._log(f"\n  === SHOWDOWN ===")

        # Evaluate all remaining hands
        results = []
        for p in active:
            all_cards = p.hole_cards + self.community_cards
            hand_rank, kickers = evaluate_hand(all_cards)
            results.append((p, hand_rank, kickers))
            self._log(
                f"  {p.name}: {p.hole_cards[0]} {p.hole_cards[1]}"
                f" — {hand_to_str(hand_rank, kickers)}"
            )

        # Sort by hand strength (best first)
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Check for ties
        best_rank = (results[0][1], results[0][2])
        winners = [r[0] for r in results if (r[1], r[2]) == best_rank]

        # Record showdown results for opponent modeling
        for p, hand_rank, kickers in results:
            won = p in winners
            # Rough hand strength from rank
            strength = hand_rank / 9.0
            self.opponent_model.get_stats(p.name).record_showdown(won, strength)

        if len(winners) == 1:
            winners[0].stack += self.pot
            self._log(f"\n  {winners[0].name} wins ${self.pot}!")
        else:
            split = self.pot // len(winners)
            remainder = self.pot % len(winners)
            names = ", ".join(w.name for w in winners)
            self._log(f"\n  Split pot! {names} each win ${split}")
            for i, w in enumerate(winners):
                w.stack += split + (1 if i < remainder else 0)

    def play_hand(self):
        """Play a complete hand of poker."""
        self.hand_number += 1
        self.community_cards = []
        self.pot = 0
        self.deck = Deck()
        self.log = []

        # Remove busted players
        self.players = [p for p in self.players if p.stack > 0]
        if len(self.players) < 2:
            self._log("Not enough players to continue.")
            return False

        for p in self.players:
            p.reset_hand()

        self._log(f"\n{'='*50}")
        self._log(f"  HAND #{self.hand_number}")
        self._log(f"  Dealer: {self.players[self.dealer_idx % len(self.players)].name}")
        self._log(f"  Blinds: ${self.small_blind}/${self.big_blind}")
        self._log(f"{'='*50}")

        # Stacks
        for p in self.players:
            self._log(f"  {p.name}: ${p.stack}")
        self._log("")

        n = len(self.players)

        # Post blinds and deal
        first_to_act = self._post_blinds()
        self._deal_hole_cards()

        # PREFLOP
        self._log(f"\n  --- Preflop ---  (Pot: ${self.pot})")
        if not self._betting_round(first_to_act, "preflop"):
            self._showdown()
            self._record_hand_stats()
            self._advance_dealer()
            return True

        self._reset_bets()

        # C-bet tracking: mark who was the preflop raiser
        # (already done in _betting_round)

        # FLOP
        self._deal_community(3, "FLOP")
        self._log(f"  Pot: ${self.pot}")
        first_postflop = (self.dealer_idx + 1) % n
        if not self._betting_round(first_postflop, "flop"):
            self._showdown()
            self._record_hand_stats()
            self._advance_dealer()
            return True

        self._reset_bets()

        # TURN
        self._deal_community(1, "TURN")
        self._log(f"  Pot: ${self.pot}")
        if not self._betting_round(first_postflop, "turn"):
            self._showdown()
            self._record_hand_stats()
            self._advance_dealer()
            return True

        self._reset_bets()

        # RIVER
        self._deal_community(1, "RIVER")
        self._log(f"  Pot: ${self.pot}")
        if not self._betting_round(first_postflop, "river"):
            self._showdown()
            self._record_hand_stats()
            self._advance_dealer()
            return True

        # SHOWDOWN
        self._showdown()
        self._record_hand_stats()
        self._advance_dealer()
        return True

    def _record_hand_stats(self):
        """Record per-hand stats for opponent modeling."""
        for p in self.players:
            stats = self.opponent_model.get_stats(p.name)
            stats.record_hand_start(
                voluntarily_entered=p.voluntarily_entered,
                raised_preflop=p.raised_preflop,
            )

    def _advance_dealer(self):
        """Move dealer button to next player with chips."""
        n = len(self.players)
        self.dealer_idx = (self.dealer_idx + 1) % n

    def play_tournament(self, max_hands: int = 100) -> Player:
        """
        Play hands until one player has all the chips
        or max_hands is reached.
        """
        self._log(f"\n{'#'*50}")
        self._log(f"  POKER TOURNAMENT")
        self._log(f"  {len(self.players)} players, ${sum(p.stack for p in self.players)} total chips")
        self._log(f"  Blinds: ${self.small_blind}/${self.big_blind}")
        self._log(f"{'#'*50}")

        for _ in range(max_hands):
            active_players = [p for p in self.players if p.stack > 0]
            if len(active_players) <= 1:
                break
            self.play_hand()

        # Final standings
        standings = sorted(self.players, key=lambda p: p.stack, reverse=True)
        self._log(f"\n{'='*50}")
        self._log(f"  FINAL STANDINGS")
        self._log(f"{'='*50}")
        for i, p in enumerate(standings, 1):
            self._log(f"  {i}. {p.name}: ${p.stack}")

        # Display opponent profiles
        self._log(f"\n{self.opponent_model.display_all()}")

        return standings[0]
