The idea is to build a language model to guess prime numbers. I'm curious what types of patterns it will learn: growing over time, no even numbers, twin primes, etc.

Currently a work in progress, very incomplete.

Using parts of Andrez Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) as a reference.

Uses [primesieve](https://github.com/kimwalisch/primesieve) from the command line to quickly generate training and testing data.

Uses [Ollama](https://github.com/ollama/ollama-python) to generate primes with LLama 3 for an interesting comparison.
