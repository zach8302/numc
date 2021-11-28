# numc

### Provide answers to the following questions.
- How many hours did you spend on the following tasks?
  - Task 1 (Matrix functions in C): 3
  - Task 2 (Speeding up matrix operations): 6
- Was this project interesting? What was the most interesting aspect about it?
  It is very exciting to watch your solution go faster than the naive one. Finding clever ways to optimize is satisfying. Finding small optimizations can feel very satisfying and clever solutions too.
- What did you learn?
  I got more comfortable using 1D matrices, vectorization and multithreading. It took a while to find a soltuion that worked for matrix multiplication, initially I tried todo many rows at once but found that if you optimize the innermost loop any future optimizations exponentiate and make it significanly faster.
- Is there anything you would change?
It could be made clearer that you are allowed to research algorithms and solutions instead of trying to bruteforce. Also it may be helpful to require mat mul be implemented in a way that doesn't require a temp matrix. This would definiteley alleviate the stress and difficulty of the power function and the memory usage.
