### If *n = no. of sample , m = no. of features , d = no. of iterations*

- The complexity for calculating y_pred = O(m*n)
- The complexity for calculating gradient = O(m*n)
- So in one iteration we take O(m\*n + m\*n) = O(m*n) time
- We do this for d iterations
### Learning/Training complexity = **O(m\*n\*d)**

- For testing we calculate the y_pred which takes = O(m*n) time
### Test/Prediction complexity = **O(m\*n)**

- We are storing X,Y and theta in memory
- Storing X takes O(m*n) , Y takes O(n) , theta takes O(m)
### Space complexity = **O(m\*n + m + n)**