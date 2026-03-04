# PageRank — Understanding the Random Surfer and the Google Matrix

When we think about the web, we should think of it as a **directed graph**:

- **Nodes** → Web pages  
- **Edges** → Hyperlinks  

The central question becomes:

> How do we rank pages based purely on link structure?

---

# Not All Pages Are Equally Important

Some pages receive millions of links. Others receive almost none.

The key insight behind PageRank is:

- Importance depends on **incoming links**
- Not all incoming links are equal
- A link from an important page should count more

This creates a recursive definition:

> A page is important if it is linked to by other important pages.

That recursive idea is the heart of PageRank.

---

# The “Flow” Model of PageRank

The PageRank equation is:

\[
r_j = \sum_{i \rightarrow j} \frac{r_i}{d_i}
\]

Where:

- \( r_j \) = rank of page \( j \)
- \( r_i \) = rank of page \( i \)
- \( d_i \) = out-degree of page \( i \)

Each page distributes its importance evenly across its outgoing links.

In matrix form:

\[
\mathbf{r} = M \mathbf{r}
\]

Where:

- \( M \) is a **column-stochastic matrix**
- Columns sum to 1
- \( r \) is the PageRank vector

This means:

> PageRank is the principal eigenvector of a stochastic matrix.

---

# The Random Surfer Interpretation

Another way to understand PageRank is probabilistic.

Imagine a random web surfer:

1. Start on a random page  
2. At each step, follow a random outgoing link  
3. Repeat forever  

Let \( p(t) \) be the probability distribution of the surfer at time \( t \).

Then:

\[
p(t+1) = M p(t)
\]

If this converges, we reach a stationary distribution:

\[
r = M r
\]

PageRank is exactly that stationary distribution.

---

# Solving PageRank with Power Iteration

We cannot directly solve \( r = M r \) for billions of pages.

Instead, we use **power iteration**:

1. Initialize:

\[
r^{(0)} = \left[ \frac{1}{N}, \dots, \frac{1}{N} \right]
\]

2. Iterate:

\[
r^{(t+1)} = M r^{(t)}
\]

3. Stop when:

\[
\| r^{(t+1)} - r^{(t)} \|_1 < \varepsilon
\]

About 50 iterations are typically enough in practice.

Why does this work?

Because repeated multiplication amplifies the dominant eigenvector and suppresses the others.

---

# Problems in the Basic Model

Two issues appear in real web graphs:

### 1. Dead Ends
Pages with no outgoing links.

Problem:
- Random walk gets stuck
- Probability mass disappears

### 2. Spider Traps
Groups of pages that only link to each other.

Problem:
- Rank accumulates inside the trap
- The rest of the web loses probability mass

These break the simple Markov chain assumptions.

---

# The Google Solution: Teleportation

To fix these issues, Google introduced **teleportation**.

At each step, the surfer:

- With probability \( \beta \): follows a link  
- With probability \( 1 - \beta \): jumps to a random page  

This gives the modified equation:

\[
r_j = \sum_{i \rightarrow j} \beta \frac{r_i}{d_i} + \frac{1-\beta}{N}
\]

Or in matrix form:

\[
\mathbf{r} = A \mathbf{r}
\]

Where:

\[
A = \beta M + (1-\beta)\frac{1}{N}\mathbf{1}
\]

This ensures:

- The Markov chain is irreducible  
- The stationary distribution exists  
- Convergence is guaranteed  

Typical choice:

\[
\beta \approx 0.85
\]

Meaning the surfer teleports roughly every 6–7 steps.

---

# Efficient Computation at Web Scale

The web has billions of pages.

Storing the full matrix is impossible:

\[
N^2 \approx 10^{18}
\]

Instead:

- Store only nonzero links (sparse matrix)
- Multiply using sparse matrix–vector products
- Add teleportation term separately

Each iteration costs:

\[
O(|M|)
\]

Where \( |M| \) is the number of links.

This makes PageRank scalable.

---

# Why Power Iteration Converges

If the largest eigenvalue is \( \lambda_1 \) and others are smaller:

\[
M^k r^{(0)} \rightarrow \lambda_1^k c_1 x_1
\]

All smaller eigenvalues shrink relative to the dominant one.

So repeated multiplication isolates the principal eigenvector.

That eigenvector is PageRank.

---

# What PageRank Actually Measures

PageRank measures:

- **Global popularity**
- Not semantic relevance
- Not topic-specific authority

Limitations:

- Biased toward high-degree nodes
- Vulnerable to link spam
- Ignores page content

Extensions include:

- Personalized PageRank
- Topic-sensitive PageRank
- HITS (Hubs and Authorities)

---

# Final Intuition

PageRank is:

- A recursive voting system  
- A stationary distribution of a Markov chain  
- The dominant eigenvector of a stochastic matrix  
- A scalable sparse linear algebra computation  

All from one simple idea:

> Important pages are linked to by other important pages.
