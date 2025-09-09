# POLICY EVALUATION

## AIM
Deploy the frozen-lake MDP. Find value function for both the policies given using policy evaluation and compare them.

## PROBLEM STATEMENT
This is an experiment in Reinforcement Learning where you compare different policies in a Frozen-Lake environment using policy evaluation.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
  prev_V=np.zeros(len(P))
  while True:
    V=np.zeros(len(P))
    for s in range(len(P)):
      for prob, next_state, reward, done in P[s][pi(s)]:
        V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))

    if np.max(np.abs(prev_V-V))<theta:
      break
    prev_V=V.copy()
  return V
```

## OUTPUT:
# Policy 1

<img width="684" height="547" alt="image" src="https://github.com/user-attachments/assets/4d64e8a5-e65f-4b11-a668-4ac3e59af3c7" />

# Policy 2

<img width="625" height="706" alt="image" src="https://github.com/user-attachments/assets/4fdbe1f4-519d-4a35-a43d-5059f5c60b7b" />

# Policy Evaluation of Policies

<img width="544" height="431" alt="image" src="https://github.com/user-attachments/assets/b1452a23-a551-43dd-a3ef-a5d82a896edf" />

# Comparing Policies

<img width="709" height="422" alt="image" src="https://github.com/user-attachments/assets/4472e53b-4b00-4735-bbde-747ade590783" />

## RESULT:
Therefore, policies are compared successfully using policy evaluation function in Frozen-Lake MDP.
