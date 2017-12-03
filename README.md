# video-classification

## Timing Benchmarks

Initial benchmarks indicate extraction of feature vector from a video takes about 3 seconds. However, this may be heavily influenced by unknown sources of latency.

Preprocessing the UCF101 dataset with 10 frames per video takes about two hours, indicating that the real time requirement is about 0.5 seconds.