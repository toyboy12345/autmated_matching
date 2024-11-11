import pulp
import numpy as np
import pickle

# Define the problem as a minimization problem
problem = pulp.LpProblem("2_Agents_Problem", pulp.LpMinimize)

# Define the variable tensors for `gs` and `ts`
gs = pulp.LpVariable.dicts("gs", ((p1, p2, q1, q2, w, f) for p1 in range(2) for p2 in range(2) for q1 in range(2) for q2 in range(2) for w in range(2) for f in range(2)), lowBound=0, upBound=1, cat="Continuous")
ts = pulp.LpVariable.dicts("ts", ((p1, p2, q1, q2, w, f) for p1 in range(2) for p2 in range(2) for q1 in range(2) for q2 in range(2) for w in range(2) for f in range(2)), lowBound=0, cat="Continuous")

# Define the objective function
problem += pulp.lpSum(ts[p1, p2, q1, q2, w, f] for p1 in range(2) for p2 in range(2) for q1 in range(2) for q2 in range(2) for w in range(2) for f in range(2))

# Preferences list
succ = [[[], [0]], [[1], []]]
succeq = [[[0], [0, 1]], [[0, 1], [1]]]
change = [[1], [0]]

# Define the constraints
for p1 in range(2):
    for p2 in range(2):
        for q1 in range(2):
            for q2 in range(2):
                ps = [p1, p2]
                qs = [q1, q2]

                # Feasibility constraints
                for w in range(2):
                    problem += pulp.lpSum(gs[p1, p2, q1, q2, w, f] for f in range(2)) <= 1
                for f in range(2):
                    problem += pulp.lpSum(gs[p1, p2, q1, q2, w, f] for w in range(2)) <= 1

                # Stability constraints
                for w in range(2):
                    for f in range(2):
                        constraint = ts[p1, p2, q1, q2, w, f] + gs[p1, p2, q1, q2, w, f]
                        constraint += sum(gs[p1, p2, q1, q2, w, f_] for f_ in succ[ps[w]][f])
                        constraint += sum(gs[p1, p2, q1, q2, w_, f] for w_ in succ[qs[f]][w])
                        problem += constraint >= 1

                # Strategy-proofness constraints
                for w in range(2):
                    for p_ in range(2):
                        for f in range(2):
                            constraint = 0
                            flag = False
                            for f_ in succeq[ps[w]][f]:
                                flag = True
                                if w == 0:
                                    constraint += gs[p_, p2, q1, q2, w, f_] - gs[p1, p2, q1, q2, w, f_]
                                elif w == 1:
                                    constraint += gs[p1, p_, q1, q2, w, f_] - gs[p1, p2, q1, q2, w, f_]
                            if flag:
                                problem += constraint <= 0

                for f in range(2):
                    for q_ in range(2):
                        for w in range(2):
                            constraint = 0
                            flag = False
                            for w_ in succeq[qs[f]][w]:
                                flag = True
                                if f == 0:
                                    constraint += gs[p1, p2, q_, q2, w_, f] - gs[p1, p2, q1, q2, w_, f]
                                elif f == 1:
                                    constraint += gs[p1, p2, q1, q_, w_, f] - gs[p1, p2, q1, q2, w_, f]
                            if flag:
                                problem += constraint <= 0

                # Anonymity constraints
                for f in range(2):
                    problem += gs[p1, p2, q1, q2, 0, f] == gs[p2, p1, change[q1][0], change[q2][0], 1, f]
                    problem += gs[p1, p2, q1, q2, 1, f] == gs[p2, p1, change[q1][0], change[q2][0], 0, f]

                for w in range(2):
                    problem += gs[p1, p2, q1, q2, w, 0] == gs[change[p1][0], change[p2][0], q2, q1, w, 1]
                    problem += gs[p1, p2, q1, q2, w, 1] == gs[change[p1][0], change[p2][0], q2, q1, w, 0]

# Solve the problem
print("Start optimizing...")
problem.solve()
problem.toJson("pulp_2agents.json")



print(problem.variables)

# Output results
# with open("log_2agents.txt", 'w') as f:
#     f.write(str(pulp.value(problem.objective)))

dic = {}
for p1 in range(2):
    for p2 in range(2):
        for q1 in range(2):
            for q2 in range(2):
                r = [[0, 0], [0, 0]]
                for w in range(2):
                    for f in range(2):
                        r[w][f] = pulp.value(gs[p1, p2, q1, q2, w, f])
                dic[f"{p1}{p2}{q1}{q2}"] = r

with open("output_2agents.pkl", "wb") as f:
    pickle.dump(dic, f)

print("Optimization complete.")
