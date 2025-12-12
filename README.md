# LLM-Driven Agent-Based Model for Flood Adaptation

This repository contains a Python-based **Agent-Based Model (ABM)** that simulates how households adapt to long-term flood risks. Unlike traditional ABMs that use static rules, this simulation employs **Generative AI (Large Language Models via Ollama)** to drive agent decision-making.

Agents behave according to **Protection Motivation Theory (PMT)**, weighing threat appraisals (severity, vulnerability) against coping appraisals (efficacy, cost) to make yearly decisions: **Do Nothing, Buy Insurance, Elevate House, or Relocate.**

## 🚀 Key Features

* **Generative Agents:** Each agent is powered by a local LLM (`gemma3:4b`), generating unique, context-aware reasoning for every decision.
* **Protection Motivation Theory (PMT):** Agents evaluate risks based on psychological factors, not just pure economic utility.
* **Dynamic Trust Logic:** Trust in insurance and neighbors acts as a state variable updated by **event-driven logic** (e.g., "Claim Hassle," "Peace of Mind," "Social Proof").
* **Verbalized Prompts:** Numerical variables (e.g., trust = 0.8) are converted into natural language (e.g., "You strongly trust...") to improve LLM roleplay.
* **Stochastic Memory:** Agents possess a sliding memory window and have a chance to randomly recall past events or forget recent ones.
* **Probabilistic Risk Reduction:** House elevation reduces flood risk by 80% (permanent physical change) rather than granting absolute immunity.
* **Robust Architecture:** Features batch processing, retry logic for LLM failures, and reproducible "warm start" capabilities via CSV profiles.
