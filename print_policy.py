import numpy as np

def print_policy(q_table, SIZE):
   # left, down, right, up
   actions = [ ' ⬅ ', ' ⬇ ', ' ➡ ', ' ⬆ ' ]

   for i, r in enumerate(q_table):
      if 0 == (i % SIZE):
         print()

      max_action = np.argmax(r)
      print(actions[max_action], end='')

