## Description

This directory contains test scripts to check if the source code for Abstraction construction, Product Construction, Strategy Synthesis, and Rolling our strategies is functioning correctly or not.
- [x] `test script(s) have been implemented`, and
- [ ] means test scripts will be implemented in future iterations.

Here is a list of available tests:

### Abstraction Construction

- [ ] `Dynamic (w human intervention) Franka Abstraction Construction`
- [ ] `Franka (w/o human intervention) Abstraction Construction`
- [ ] `Minigrid Abstraction Construction`

### Product Construction

- [ ] `Franka Product Construction w LTL, LTLf DFAs`
- [ ] `Minigrid Product Construction w LTL, LTLf DFAs`

### Strategy Synthesis

#### Qualitative algorithms

- [ ] `Adversarial Game (Qualitative Algo.) Strategy Synthesis for Franka Abstraction`
- [ ] `Adversarial Game (Qualitative Algo.) Strategy Synthesis for Minigrid Abstraction`
- [ ] `Cooperative Game (Qualitative Algo.) Strategy Synthesis for Franka Abstraction`
- [ ] `Cooperative Game (Qualitative Algo.) Strategy Synthesis for Minigrid Abstraction`
- [ ] `Qualitative Best-Effort Strategy Synthesis (w and w/o realizable tasks) for Franka Abstraction for LTL and LTLf specifications`
- [ ] `Qualitative Best-Effort Strategy Synthesis (w and w/o realizable tasks) for Minigrid Abstraction for LTL and LTLf specifications`

#### Quantitative algorithms

- [ ] `Quantitative Best-Effort Strategy Synthesis (w and w/o realizable tasks) for Franka Abstraction for LTL and LTLf specifications`
- [ ] `Quantitative Best-Effort Strategy Synthesis (w and w/o realizable tasks) for Minigrid Abstraction for LTL and LTLf specifications`
- [ ] `Qualitative Safe Reach Best-Effort Strategy Synthesis (w and w/o realizable tasks) for Franka Abstraction for LTL and LTLf specifications`
- [ ] `Qualitative Safe Reach Best-Effort Strategy Synthesis (w and w/o realizable tasks) for Minigrid Abstraction for LTL and LTLf specifications`
- [ ] `Regret-Minimizing strategy synthesis for Franka Abstraction`
- [ ] `Regret-Minimizing strategy synthesis for Minigrid Abstraction`

### Test Packages

To run each test package, use the following command

```bash
python3 -m unittest discover -s tests.<directory-name> -bv
```

The `-s` flag allows you to specify directory to start discovery from. Use only `-b` if you want to suppress all the prints (including progress). Use `-v` for verbose print or `-bv` to just print if a test failed or pass.


### Test Scripts

To run the test scripts within each package, use the following command to run one module

```bash
cd <root/of/project>

python3 -m tests.<directory-name>.<module-nane> -b
```

The `-m` flag runs the test as a module while `-b` flag suppresses the output of the code if all the tests pass (you can also use `-bv`). If the test fails, then it throws the error message at the very beginning and then the rest of the output. 

### Test everything

To run all the tests use the following command

```bash
python3 -m unittest -bv
```