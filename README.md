# Hold'em Bot

An autonomous No-Limit Texas Hold'em poker bot that uses Monte Carlo simulation, counterfactual regret minimization (CFR), and opponent modeling to make decisions. Built in pure Python with zero external dependencies.

Watch AI players with distinct personalities compete against each other in real-time from your terminal — and watch them learn to exploit each other.

## How It Works

The bot uses a two-tier decision architecture inspired by professional poker AIs like Libratus and Pluribus:

**Tier 1 — Blueprint Strategy (CFR):** Before playing, the engine can train a baseline strategy via Monte Carlo Counterfactual Regret Minimization (MCCFR). This is the same family of algorithms that solved heads-up limit Hold'em and powered the first superhuman no-limit bots. The engine plays millions of hands against itself, tracks how much it "regrets" not taking each action, and converges toward a Nash equilibrium — the theoretically unexploitable strategy.

**Tier 2 — Real-Time Exploitation:** During live play, the bot layers exploitative adjustments on top of the blueprint. It tracks each opponent's VPIP, PFR, aggression factor, fold-to-bet frequency, and showdown tendencies, then classifies them (nit, TAG, LAG, calling station, maniac) and adjusts accordingly. Bluff the nit. Value-bet the calling station. Trap the maniac.

**Monte Carlo Equity:** When no CFR data exists for a given situation, the bot falls back to pure Monte Carlo simulation — dealing out random board completions hundreds of times, evaluating each at showdown, and returning a win probability. It then compares equity against pot odds to make the mathematically sound play.

### Decision Flow

```
Decision Point
    │
    ├── CFR strategy exists? ──→ Use blueprint + apply opponent adjustments
    │
    └── No CFR data? ──→ Monte Carlo equity vs pot odds
                              │
                              ├── Board texture analysis (wet/dry/paired)
                              ├── SPR-aware commitment decisions
                              ├── Position-adjusted aggression
                              └── Opponent-specific exploits
```

### Three Playing Styles

- **Tight** — Patient, selective, waits for strong hands. Rarely bluffs.
- **Balanced** — Solid fundamentally sound play. Moderate aggression.
- **Aggressive** — Wide opening range, frequent raises, higher bluff frequency.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/ethangarofalo/holdem-bot.git
cd holdem-bot

# Run a 6-player tournament
python play.py

# Train CFR strategy first, then play
python play.py --train 5000

# Customize the game
python play.py --players 4 --stack 500 --hands 50

# Fast mode with opponent profiles at the end
python play.py --speed fast --profiles

# Quiet mode (only final results + profiles)
python play.py --quiet --hands 200

# Train and save strategy, then load it later
python cfr.py                                    # trains + saves to strategy.json
python play.py --load-strategy strategy.json     # play using trained strategy
```

No dependencies to install. Pure Python 3.8+.

## Architecture

```
holdem-bot/
├── cards.py          # Card, Deck, and hand evaluation engine
├── monte_carlo.py    # Monte Carlo equity simulation
├── info_sets.py      # Information set abstraction (game state bucketing)
├── cfr.py            # Counterfactual Regret Minimization trainer
├── opponent_model.py # Live opponent tracking and exploitation
├── bot.py            # Decision engine (the brain)
├── table.py          # Game table, betting rounds, showdown
├── play.py           # CLI entry point for tournaments
└── README.md
```

**`cards.py`** — Represents cards, decks, and evaluates poker hands. Finds the best 5-card hand from any 5-7 cards using combinatorial evaluation. Handles all standard hand rankings from high card through royal flush, including the ace-low straight (wheel).

**`monte_carlo.py`** — The simulation engine. Given your hole cards and any known community cards, it deals random completions thousands of times and counts how often you win. Returns win/tie/loss percentages and overall equity.

**`info_sets.py`** — Information set abstraction, the key insight from game-theoretic poker AI. Groups the infinite space of poker situations into tractable buckets: hand strength (6 levels), board texture (5 types), stack-to-pot ratio (3 tiers), position, and action facing. This makes CFR training feasible without sacrificing strategic nuance.

**`cfr.py`** — The MCCFR training engine. Plays hands against itself, computing regret for each decision at each information set, then updates strategy proportional to positive regret (regret matching). Over thousands of iterations, converges toward Nash equilibrium. Strategies can be saved/loaded as JSON.

**`opponent_model.py`** — Tracks real-time opponent statistics: VPIP, PFR, aggression factor, fold-to-bet rate, continuation bet frequency, 3-bet frequency, and showdown results. Classifies opponents into archetypes (nit, TAG, LAG, calling station, maniac) and returns specific exploitative adjustments.

**`bot.py`** — The decision engine. Queries CFR for the blueprint strategy, applies opponent-specific adjustments, and falls back to heuristic Monte Carlo play when needed. Incorporates board texture and SPR into bet sizing (geometric sizing on wet boards, smaller bets on dry boards).

**`table.py`** — Runs the actual poker game. Manages the deck, blinds, dealing, four betting rounds, and showdown. Supports 2-8 players, handles all-in situations and split pots, and feeds every action into the opponent model.

**`play.py`** — The tournament runner. Creates a table of bots with different personalities and lets them compete. Supports CFR pre-training, strategy loading, and displays opponent profiles alongside the final leaderboard.

## Poker Theory: Monte Carlo vs CFR

This project implements two fundamentally different approaches to poker AI, and understanding the distinction is the key to understanding modern poker theory.

### Monte Carlo (v1 approach)
Monte Carlo estimates **hand equity** — the probability of winning if all remaining cards were dealt randomly. It answers: "Given what I know right now, how often do I win?" This is powerful for single-decision analysis but has a critical limitation: it treats poker as a game of cards rather than a game of decisions. It doesn't account for how opponents will react to your bets, or how your betting patterns reveal information about your hand.

### CFR (v2 approach, inspired by fedden/poker_ai)
Counterfactual Regret Minimization treats poker as what it actually is: an **imperfect information game** where strategy depends on what your opponents think you have. CFR doesn't ask "what are my odds?" — it asks "what would I regret not doing?" By iterating through millions of self-play hands, it builds a strategy that considers the entire game tree: if I raise here, my opponent will fold X% of the time, call Y%, and re-raise Z%, and each of those branches leads to further decisions. The result converges to a Nash equilibrium — a strategy that cannot be exploited regardless of what opponents do.

### Why Both?
Pure GTO (game-theory optimal) play is unexploitable but doesn't maximize profit against weak opponents. Pure exploitation is profitable against bad players but vulnerable to counter-exploitation. The best approach — and what this bot implements — is a **GTO baseline with exploitative adjustments**: play the theoretically sound strategy by default, then deviate when you identify specific opponent weaknesses.

## Monte Carlo in Action

```python
from cards import Card
from monte_carlo import estimate_equity

# Pocket aces vs 1 opponent preflop
hand = [Card.from_str("Ah"), Card.from_str("As")]
result = estimate_equity(hand, [], num_opponents=1, simulations=2000)
print(f"Equity: {result['equity']:.1%}")  # ~85%

# Flush draw on the flop
hand = [Card.from_str("Ah"), Card.from_str("Kh")]
board = [Card.from_str("7h"), Card.from_str("2h"), Card.from_str("Tc")]
result = estimate_equity(hand, board, num_opponents=1, simulations=2000)
print(f"Equity: {result['equity']:.1%}")  # ~55%
```

## CFR Training

```python
from cfr import CFRTrainer

# Train a strategy through self-play
trainer = CFRTrainer()
trainer.train(iterations=10000)
trainer.strategy.save("strategy.json")

# The trained strategy maps information sets to action probabilities
# Example: facing a bet on the flop with a strong hand in late position
# → raise 45%, call 40%, fold 15%
```

## What This Demonstrates

This project is a practical exercise in probability, game theory, and decision-making under uncertainty. It implements the same algorithmic framework — Monte Carlo simulation layered with CFR and opponent modeling — that powers the AIs that have beaten the best human poker players. The bot faces the same fundamental problem that any poker player faces: you must commit resources based on incomplete information, with the quality of your decisions compounding over hundreds of hands.

The progression from pure Monte Carlo (v1) to CFR + exploitation (v2) mirrors the evolution of poker AI itself: from "calculate your odds" to "calculate your odds, anticipate your opponent's strategy, and find the response they can't exploit."

## Future Ideas

- **Hand history export** — Save games in standard poker hand history format
- **Human vs. bot mode** — Play against the bots yourself from the terminal
- **Deep CFR** — Train a neural network to approximate the CFR strategy
- **Real-time search** — Depth-limited lookahead at decision points (Libratus-style)
- **Web interface** — Visualize the game with a card table UI

## About

Built by [Ethan Garofalo](https://ethangarofalo.github.io/ethangarofalo) — poker player, writer, and builder at the intersection of philosophy and AI.
