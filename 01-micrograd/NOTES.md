# Day 1 — micrograd (Fri Aug 14)

Video: "The spelled-out intro to neural networks and backpropagation: building micrograd" (2h25m)

Deliverable: my own autograd engine, from memory.
- value.py — Value class: +, *, tanh/relu, backward()
- test it: gradient through a*b + c matches manual calculus

Day ends when MY backprop chains a gradient correctly.

## Honest accounting (end of Day 1)
- From memory: Value class core; _backward closures (struggled on add, referenced after real struggle — rule followed)
- Copied, and that's fine: imports, draw_dot/trace viz, constants
- Referenced, asterisk: topological sort — not yet produced cold
- Day 2 warm-up: blank cell, rewrite backward() + topo sort from memory. That clears the asterisk or finds the gap.
