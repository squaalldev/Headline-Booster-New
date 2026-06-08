# Field Notes

## Problem

Many entrepreneurs struggle to write clear headlines because they try to explain too much, sound generic, or do not connect the offer to the audience's desired result.

## Small model angle

Headline generation is a narrow, structured task. A tiny model can perform well if the input is constrained to four pieces of information and the output format is fixed. The first real-model target is Qwen/Qwen2.5-1.5B-Instruct, with Qwen/Qwen2.5-3B-Instruct as an under-4B quality fallback.

## Design decision

The app asks only four questions and focuses on one job: generating headlines.

## Future work

Test Qwen/Qwen2.5-1.5B-Instruct on ZeroGPU, compare it against Qwen/Qwen2.5-3B-Instruct, and keep the smallest model that produces strong Spanish headline quality.
