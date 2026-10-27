"""The investment assistant: a buyer's questions about one listed property, answered with numbers.

A visitor inside a tour asks the question every listing invites — "is this a good buy?" — and
the honest answer is not a verdict but three kinds of number, kept apart: what the agent says
about the flat (price, rent, aidat), what the market says about the area (rents, price per m²,
mortgage rates — found on the web, with a link and a date), and what follows from the two
(yield, payback, purchase costs, the instalment). This package produces exactly that, and the
three-way split is the design rather than a formatting choice:

``calc``      Every computed figure. Pure functions, hand-checked in the tests, no model in
              sight. The model is NEVER allowed to do arithmetic — not a sum, not a
              percentage — because a fluent paragraph with a wrong yield in it reads exactly
              as well as one with the right yield.
``tools``     The calculators as strict client tools the model calls. The agent's own figures
              are filled in on this side, from the property, so the model cannot mistype the
              asking price; it passes only what genuinely varies (a scenario price, a market
              rent it found, a benchmark with its source).
``rates``     Turkey's purchase costs and the example mortgage terms — tapu harcı, döner
              sermaye, commission, KDV, LTV caps — as configurable, dated, sourced values.
              Never silently hard-coded: the dashboard shows them and every tool result
              echoes the ones it used.
``facts``     The property as the model is told about it: one frozen text block per
              conversation, so the history stays append-only and the prompt cache holds.
``advisor``   The streaming chat loop against the Messages API: web search for market data,
              the calculators as tools, refusals, pause_turn, cost accounting per call.
``store``     Conversations (in memory, never on disk — a visitor's questions are not ours to
              keep) and the per-call usage rows the budget and the dashboard read.
``ratelimit`` Per-IP and per-property windows. In memory, keyed by a salted digest: the
              privacy policy says the product keeps no visitor IP, and it keeps none here.
``audit``     The bench's checker: every number in an answer must trace to a tool result, the
              agent's inputs or a cited source.

What this package is not: an adviser. It says what the numbers show and never "buy" or "don't".
"""
