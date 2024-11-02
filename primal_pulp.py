import pulp
import numpy as np
import pickle

# Define the problem as a minimization problem
problem = pulp.LpProblem("6_Agents_Problem", pulp.LpMinimize)

# Define variable tensors for `gs` and `ts`
gs = pulp.LpVariable.dicts("gs", ((p1, p2, p3, q1, q2, q3, w, f) for p1 in range(6) for p2 in range(6) for p3 in range(6) for q1 in range(6) for q2 in range(6) for q3 in range(6) for w in range(3) for f in range(3)), lowBound=0, upBound=1, cat="Continuous")
ts = pulp.LpVariable.dicts("ts", ((p1, p2, p3, q1, q2, q3, w, f) for p1 in range(6) for p2 in range(6) for p3 in range(6) for q1 in range(6) for q2 in range(6) for q3 in range(6) for w in range(3) for f in range(3)), lowBound=0, cat="Continuous")

# Define the objective function
problem += pulp.lpSum(ts[p1, p2, p3, q1, q2, q3, w, f] for p1 in range(6) for p2 in range(6) for p3 in range(6) for q1 in range(6) for q2 in range(6) for q3 in range(6) for w in range(3) for f in range(3))

# Preferences list
succ = [
    [[], [0], [0, 1]],
    [[], [0, 2], [0]],
    [[1], [], [0, 1]],
    [[2], [0, 2], []],
    [[1, 2], [], [1]],
    [[1, 2], [2], []]
]
succeq = [
    [[0], [0, 1], [0, 1, 2]],
    [[0], [0, 1, 2], [0, 2]],
    [[0, 1], [1], [0, 1, 2]],
    [[0, 2], [0, 1, 2], [2]],
    [[0, 1, 2], [1], [1, 2]],
    [[0, 1, 2], [1, 2], [2]]
]
change = [
    [2, 5, 1],
    [4, 3, 0],
    [0, 4, 3],
    [5, 1, 2],
    [1, 2, 5],
    [3, 0, 4]
]

# Define constraints
for p1 in range(6):
    for p2 in range(6):
        for p3 in range(6):
            for q1 in range(6):
                for q2 in range(6):
                    for q3 in range(6):
                        ps = [p1, p2, p3]
                        qs = [q1, q2, q3]

                        # Feasibility constraints
                        for w in range(3):
                            problem += pulp.lpSum(gs[p1, p2, p3, q1, q2, q3, w, f] for f in range(3)) <= 1
                        for f in range(3):
                            problem += pulp.lpSum(gs[p1, p2, p3, q1, q2, q3, w, f] for w in range(3)) <= 1

                        # Stability constraints
                        for w in range(3):
                            for f in range(3):
                                constraint = ts[p1, p2, p3, q1, q2, q3, w, f] + gs[p1, p2, p3, q1, q2, q3, w, f]
                                for f_ in succ[ps[w]][f]:
                                    constraint += gs[p1, p2, p3, q1, q2, q3, w, f_]
                                for w_ in succ[qs[f]][w]:
                                    constraint += gs[p1, p2, p3, q1, q2, q3, w_, f]
                                problem += constraint >= 1

                        # Strategy-proofness constraints
                        for w in range(3):
                            for p_ in range(6):
                                for f in range(3):
                                    constraint = 0
                                    for f_ in succeq[ps[w]][f]:
                                        if w == 0:
                                            constraint += gs[p_, p2, p3, q1, q2, q3, w, f_] - gs[p1, p2, p3, q1, q2, q3, w, f_]
                                        elif w == 1:
                                            constraint += gs[p1, p_, p3, q1, q2, q3, w, f_] - gs[p1, p2, p3, q1, q2, q3, w, f_]
                                        elif w == 2:
                                            constraint += gs[p1, p2, p_, q1, q2, q3, w, f_] - gs[p1, p2, p3, q1, q2, q3, w, f_]
                                    problem += constraint <= 0

                        for f in range(3):
                            for q_ in range(6):
                                for w in range(3):
                                    constraint = 0
                                    for w_ in succeq[qs[f]][w]:
                                        if f == 0:
                                            constraint += gs[p1, p2, p3, q_, q2, q3, w_, f] - gs[p1, p2, p3, q1, q2, q3, w_, f]
                                        elif f == 1:
                                            constraint += gs[p1, p2, p3, q1, q_, q3, w_, f] - gs[p1, p2, p3, q1, q2, q3, w_, f]
                                        elif f == 2:
                                            constraint += gs[p1, p2, p3, q1, q2, q_, w_, f] - gs[p1, p2, p3, q1, q2, q3, w_, f]
                                    problem += constraint <= 0

                        # Anonymity constraints
                        for f in range(3):
                            problem += gs[p1, p2, p3, q1, q2, q3, 0, f] == gs[p2, p1, p3, change[q1][0], change[q2][0], change[q3][0], 1, f]
                            problem += gs[p1, p2, p3, q1, q2, q3, 1, f] == gs[p2, p1, p3, change[q1][0], change[q2][0], change[q3][0], 0, f]
                            problem += gs[p1, p2, p3, q1, q2, q3, 2, f] == gs[p2, p1, p3, change[q1][0], change[q2][0], change[q3][0], 2, f]
                            problem += gs[p1, p2, p3, q1, q2, q3, 0, f] == gs[p3, p2, p1, change[q1][1], change[q2][1], change[q3][1], 2, f]
                            problem += gs[p1, p2, p3, q1, q2, q3, 2, f] == gs[p3, p2, p1, change[q1][1], change[q2][1], change[q3][1], 0, f]
                            problem += gs[p1, p2, p3, q1, q2, q3, 1, f] == gs[p3, p2, p1, change[q1][1], change[q2][1], change[q3][1], 1, f]

                        for w in range(3):
                            problem += gs[p1, p2, p3, q1, q2, q3, w, 0] == gs[change[p1][0], change[p2][0], change[p3][0], q2, q1, q3, w, 1]
                            problem += gs[p1, p2, p3, q1, q2, q3, w, 1] == gs[change[p1][0], change[p2][0], change[p3][0], q2, q1, q3, w, 0]
                            problem += gs[p1, p2, p3, q1, q2, q3, w, 2] == gs[change[p1][0], change[p2][0], change[p3][0], q2, q1, q3, w, 2]

print("Start optimizing...")
problem.solve()

# Output results
with open("log_pulp.txt", 'w') as f:
    f.write(str(pulp.value(problem.objective)))

dic = {}
for p1 in range(6):
    for p2 in range(6):
        for p3 in range(6):
            for q1 in range(6):
                for q2 in range(6):
                    for q3 in range(6):
                        r = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                        for w in range(3):
                            for f in range(3):
                                r[w][f] = pulp.value(gs[p1, p2, p3, q1, q2, q3, w, f])
                        dic[f"{p1}{p2}{p3}{q1}{q2}{q3}"] = r

with open("output_pulp.pkl", "wb") as f:
    pickle.dump(dic, f)

print("Optimization complete.")
