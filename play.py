#!/usr/bin/env python3
"""
Play a poker tournament between AI bots.
Watch them compete in real time from your terminal.

Usage:
    python play.py                    # Default 6-player tournament
    python play.py --players 4        # 4-player game
    python play.py --hands 50         # Limit to 50 hands
    python play.py --stack 500        # Starting stack of 500
    python play.py --quiet            # Only show final results
    python play.py --speed fast       # Skip hand-by-hand pauses
    python play.py --train 5000       # Train CFR strategy first
    python play.py --profiles         # Show opponent profiles at end
"""

import argparse
import time
import sys
from bot import PokerBot
from table import Table, Player
from opponent_model import OpponentModel
from cfr import CFRStrategy, CFRTrainer


# Bot personalities — each plays differently
BOT_CONFIGS = [
    ("Artemis", "aggressive"),
    ("Brutus", "tight"),
    ("Cassandra", "balanced"),
    ("Dante", "aggressive"),
    ("Echo", "tight"),
    ("Fortuna", "balanced"),
    ("Gaius", "aggressive"),
    ("Hector", "balanced"),
]


def create_players(num_players: int, starting_stack: int,
                   opponent_model: OpponentModel,
                   cfr_strategy: CFRStrategy = None) -> list:
    """Create a table of bot players with diverse styles."""
    configs = BOT_CONFIGS[:num_players]
    players = []
    for name, style in configs:
        bot = PokerBot(
            name=name, style=style,
            opponent_model=opponent_model,
            cfr_strategy=cfr_strategy,
        )
        players.append(Player(bot=bot, stack=starting_stack))
    return players


def main():
    parser = argparse.ArgumentParser(description="Texas Hold'em AI Tournament")
    parser.add_argument("--players", type=int, default=6,
                        help="Number of players (2-8, default: 6)")
    parser.add_argument("--hands", type=int, default=100,
                        help="Maximum hands to play (default: 100)")
    parser.add_argument("--stack", type=int, default=1000,
                        help="Starting chip stack (default: 1000)")
    parser.add_argument("--sb", type=int, default=5,
                        help="Small blind (default: 5)")
    parser.add_argument("--bb", type=int, default=10,
                        help="Big blind (default: 10)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only show final results")
    parser.add_argument("--speed", choices=["slow", "normal", "fast"],
                        default="normal",
                        help="Playback speed (default: normal)")
    parser.add_argument("--train", type=int, default=0,
                        help="Train CFR strategy for N iterations before playing")
    parser.add_argument("--load-strategy", type=str, default=None,
                        help="Load a pre-trained CFR strategy from file")
    parser.add_argument("--profiles", action="store_true",
                        help="Show detailed opponent profiles at the end")
    args = parser.parse_args()

    num_players = max(2, min(8, args.players))

    print(r"""
    ╔═══════════════════════════════════════════╗
    ║       TEXAS HOLD'EM AI TOURNAMENT         ║
    ║     Monte Carlo + CFR Poker Engine        ║
    ╚═══════════════════════════════════════════╝
    """)

    # CFR Strategy
    cfr_strategy = CFRStrategy()
    if args.load_strategy:
        if cfr_strategy.load(args.load_strategy):
            print(f"  Loaded strategy from {args.load_strategy}")
            print(f"  {cfr_strategy.stats()}")
        else:
            print(f"  Could not load {args.load_strategy}, starting fresh")

    if args.train > 0:
        print()
        trainer = CFRTrainer(strategy=cfr_strategy)
        trainer.train(iterations=args.train)
        cfr_strategy.save("strategy.json")
        print(f"  Strategy saved to strategy.json")
        print()

    # Shared opponent model
    opponent_model = OpponentModel()

    players = create_players(num_players, args.stack, opponent_model, cfr_strategy)

    print(f"  Players: {num_players}")
    print(f"  Starting Stack: ${args.stack}")
    print(f"  Blinds: ${args.sb}/${args.bb}")
    print(f"  Max Hands: {args.hands}")
    if cfr_strategy.iterations > 0:
        print(f"  CFR Strategy: {cfr_strategy.iterations} iterations")
    print()

    for p in players:
        print(f"  {p.bot.name:12s} — {p.bot.style:12s} — ${p.stack}")
    print()

    if args.speed == "slow":
        delay = 1.5
    elif args.speed == "fast":
        delay = 0.0
    else:
        delay = 0.3

    if not args.quiet:
        input("  Press Enter to start the tournament...\n")

    table = Table(
        players=players,
        small_blind=args.sb,
        big_blind=args.bb,
        verbose=not args.quiet,
    )

    # Play the tournament
    for hand_num in range(args.hands):
        active = [p for p in table.players if p.stack > 0]
        if len(active) <= 1:
            break

        table.play_hand()

        if not args.quiet and delay > 0:
            time.sleep(delay)

    # Final results
    standings = sorted(table.players, key=lambda p: p.stack, reverse=True)

    print(f"\n{'='*50}")
    print(f"  TOURNAMENT COMPLETE — {table.hand_number} hands played")
    print(f"{'='*50}")
    print()

    for i, p in enumerate(standings, 1):
        bar_len = int(p.stack / max(s.stack for s in standings) * 30) if standings[0].stack > 0 else 0
        bar = "█" * bar_len
        style_tag = f"({p.bot.style})"
        if i == 1 and p.stack > 0:
            print(f"  🏆 {i}. {p.name:12s} {style_tag:14s} ${p.stack:>6d}  {bar}")
        elif p.stack > 0:
            print(f"     {i}. {p.name:12s} {style_tag:14s} ${p.stack:>6d}  {bar}")
        else:
            print(f"  💀 {i}. {p.name:12s} {style_tag:14s} ${p.stack:>6d}  ELIMINATED")

    print()
    if standings[0].stack > 0:
        print(f"  Winner: {standings[0].name} with ${standings[0].stack}!")
    print()

    # Show opponent profiles
    if args.profiles or args.quiet:
        print(opponent_model.display_all())
        print()


if __name__ == "__main__":
    main()
