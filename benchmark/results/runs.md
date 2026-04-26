# Collocated conditions (single L4, no disaggregation):

1. **collocated_20260426_044047** — Collocated with prefix caching enabled, original 5 prompts (P1-P5)
2. **collocated_cold_20260426_044558** — Collocated with --no-enable-prefix-caching, original 5 prompts (P1-P5)
3. **collocated_cold_extended_20260426_050838** — Collocated with --no-enable-prefix-caching, full 7 prompts (P1-P7, includes P6_long 404 tokens and P7_vlong 654 tokens)
4. **collocated_prefix_cache_20260426_051527** — Collocated with prefix caching enabled, full 7 prompts (P1-P7)

# LMCache disaggregated conditions (L4 prefill + L4 decode, Redis over Tailscale):

1. **lmcache_20260426_040713** — LMCache warm cache (no Redis flush between prompts), original 5 prompts (P1-P5)
2. **lmcache_cold_20260426_043507** — LMCache cold cache (Redis flushed before each prompt), original 5 prompts (P1-P5)
3. **lmcache_cold_extended_20260426_050223** — LMCache cold cache (Redis flushed before each prompt), full 7 prompts (P1-P7)

# For your analysis, the most meaningful comparisons are:

1. **collocated_cold_extended** vs **lmcache_cold_extended** — apples to apples, both cold, both 7 prompts, this is your crossover threshold story
2. **collocated_prefix_cache** vs **lmcache_cold_extended** — best case collocated vs disaggregated, shows the ceiling of what collocated can achieve