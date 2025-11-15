# FaaSInfer Profiling Module
#
# This module extends Viduru's profiling methodology to serverless/FaaS-based
# LLM inference systems. It profiles additional components beyond traditional
# LLM serving:
#
# 1. Model Loading - Checkpoint loading times (cold start overhead)
# 2. Storage I/O - Multi-tier storage performance profiling
# 3. Scaling - Instance spin-up/spin-down timing
# 4. Migration - Live model migration costs
