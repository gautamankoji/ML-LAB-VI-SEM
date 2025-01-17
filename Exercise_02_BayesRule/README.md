# Exercise 02

## Apply Bayes’ rule in Python to solve the problem: The probability that it is Friday and that a student is absent is 3%. Since there are 5 school days in a week, the probability that it is Friday is 20%. What is the probability that a student is absent given that today is Friday? (Ans: 15%)

### Aim  

To calculate the probability of a student being absent given that today is Friday using Bayes’ theorem.  

### Theory

Bayes’ theorem provides a way to calculate the conditional probability of an event based on prior knowledge of conditions that might be related to the event. The formula is:  

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Where:

- `P(A|B)`: Probability of event A (student absent) given event B (today is Friday).  
- `P(B|A)`: Probability of event B given event A.  
- `P(A)`: Probability of event A.  
- `P(B)`: Probability of event B.  

### Procedure/Program  

```python
P_Friday_and_Absent = 0.03  # P(Friday and Absent)
P_Friday = 0.20             # P(Friday)

# calculate P(Absent | Friday) using Bayes' theorem
P_Absent_given_Friday = P_Friday_and_Absent / P_Friday

print(f"The probability that a student is absent given that today is Friday: {P_Absent_given_Friday * 100:.2f}%")
```

### Output/Explanation  

Output:
  
The program outputs:  

```bash
The probability that a student is absent given that today is Friday: 15.00%
```

Explanation:  

- The program calculates `P(Absent|Friday) = (P(Friday and Absent)) / P(Friday)`.  
- Given `P(Friday and Absent) = 0.03` and `P(Friday) = 0.20`, we get:  

$P(\text{Absent|Friday}) = \frac{0.03}{0.20} = 0.15$

- The probability of a student being absent on a Friday is thus **15%**.
