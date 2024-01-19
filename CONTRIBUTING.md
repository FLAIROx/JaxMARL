# Contributing to JaxMARL

Please help build JaxMARL into the best possible tool for the MARL community. 

## Contributing code

We actively welcome your contributions!
 - If adding an environment or algorithm, check with us that it is the right fit for the repo.
 - Fork the repo and create your branch from main.
 - Add tests, or show proof that the environment/algorithm works. The exact requirements are listed below.
 - Add a README explaining your environment/algorithm.

**Environment Requirements**
 - Unit tests (in `pytest` format) demonstrating correctness. If applicable, show correspondence to existing implementations. If transitions match, write a unit test to demonstrate this ([example](https://github.com/FLAIROx/JaxMARL/blob/be9fe46e52a736f8dd766acf98b4e0803f199dd2/tests/mpe/test_mpe.py)).
 - Training results for IPPO and MAPPO over 20 seeds, with configuration files saved to `baselines`.

**Algorithm Requirements**
 - Performance results on at least 3 environments (e.g. SMAX, MABrax & Overcooked) with at least 20 seeds per result.
 - If applicable, compare performance results to existing implementations to demonstrate correctness.

## Bug reports

We use Github's issues to track bugs, just open a new issue! Great Bug Reports tend to have:
 - A quick summary and/or background
 - Steps to reproduce (Be specific and give example code if you can)
 - What you expected would happen
 - What actually happens
 - Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)


## License 

All contributions will fall under the project's original license.

## Roadmap

Some improvements we would like to see implemented:
- [ ] improved RNN implementations. In the current implementation, the hidden size is dependent on "NUM_STEPS", it should be made independent. Speed could also be improved with an S5 architecture.
