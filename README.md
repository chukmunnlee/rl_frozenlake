# NUS-ISS StackUp Webinar on Reinforcement Learning

Event held on 16 April 2020, conducted by Chuk Munn Lee.
Event details [here](https://www.iss.nus.edu.sg/community/events/event-details/2020/04/16/default-calendar/nus-iss-stackup-webinar-an-introduction-to-reinforcement-learning?fbclid=IwAR1MasM82kObCQ9aF2PF8W83f_xYJW79aTY_NQqTKjE5O1lnaCDEY_rEAyc), Slides [here](bit.ly/stackup_intro_to_rl).
Notes taken by [@lyqht](https://github.com/lyqht)


## Setup Instructions

Set up virtual environment

```bash
python3 -m venv .venv
```

Activate virtual environment and install necessary packages

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Simple program to start the agent can be found in `simple_agent.py`

Main program implementing the monte carlo algorithm with the agent can be found in `main.py`, with many comments for explanation of what is going on.

## Notes

### Programmed vs Learnt

- **programmed**: with an input go through if else decisions
- **learnt**: find a function through machine learning that approximates decisions

### Types of machine learning problems

1. Supervised Learning
   - Input: sample data and answer
   - Algorithm: infer rules from the input
   - Goal: Find known answers/patterns.
2. Unsupervised Learning
   - Input: sample data only
   - Algorithm: use some measurement to infer similarity by grouping them
   - Goal: Find unknown patterns
3. **Reinforcement Learning**
   - Input: nth! no sample data or answer.
   - Algorithm: Infer rules (positive/negative feedback)
   - Goal: Find optimal action for every state

### Model of Reinforcement Learning

**Parameters**

- _S_: Observation
- _A_: Action
- _R_: Reward
- Environment (often presented as a grid world)
- State

**Epsilon-greedy policy**

![](https://www.oreilly.com/library/view/hands-on-reinforcement-learning/9781788836524/assets/9ae532cc-2655-4fd0-bcd2-545afc27a1df.png)
*Image from [Hands-On Reinforcement Learning with Python by Sudharsan Ravichandiran](https://www.oreilly.com/library/view/hands-on-reinforcement-learning/9781788836524/0c14fb24-1926-4cc3-8bf6-818cae23bde2.xhtml)*
