# numpy_math-mastery
#  NumPy Learning & Practice Summary

This repository documents my journey learning and practicing **NumPy**, the fundamental package for scientific computing in Python. It includes my **study notes**, **concept explanations**, and **hands-on coding challenges** covering all the core and advanced features of NumPy.

---

##  Topics Covered

###  1. Arrays Creation & Basic Info
- `np.array()`, `np.arange()`, `np.linspace()`, `np.ones()`, `np.zeros()`, `np.full()`
- Array properties: `.shape`, `.ndim`, `.size`, `.dtype`

###  2. Indexing & Slicing
- Accessing and modifying values in 1D and 2D arrays
- Fancy indexing and boolean indexing

### 3. Array Operations
- Element-wise operations: `+`, `-`, `*`, `/`, `**`
- Logical operations: `>`, `<`, `==`, `!=`, `np.where`, `np.any`, `np.all`

###  4. Broadcasting
- Understanding rules for broadcasting
- Using broadcasting with scalars and arrays of different shapes

###  5. Shape Manipulation
- Reshape: `.reshape()`, `.ravel()`, `.flatten()`
- Transpose: `.T`
- Add dimensions: `np.newaxis`, `.reshape(1, -1)`, `.reshape(-1, 1)`

### 6. Combining and Splitting Arrays
- Vertical and horizontal stacking: `np.vstack()`, `np.hstack()`
- Concatenation: `np.concatenate()`
- Splitting: `np.split()`, `np.array_split()`

###  7. Masking and Filtering
- Using boolean masks to filter arrays
- Conditional replacement: `np.where(condition, x, y)`

###  8. Mathematical Functions
- `np.sum()`, `np.mean()`, `np.std()`, `np.max()`, `np.min()`, `np.argmin()`, `np.argmax()`

###  9. Special Ranges
- `np.linspace(start, end, num)`

###  10. Handling Missing Data (NaN)
- `np.isnan()`, `np.nanmean()`, `np.nansum()`, `np.nan_to_num()`

###  11. Views vs Copies
- `.view()` creates a linked view of the data
- `.copy()` creates an independent copy

### 12. Linear Algebra & Matrix Ops
- Matrix multiplication: `np.dot(a, b)`, `a @ b`
- Inverse: `np.linalg.inv()`
- Determinant: `np.linalg.det()`
- Eigenvalues & eigenvectors: `np.linalg.eig() → (eigenvalues, eigenvectors)`

---

##  Advanced Practice Challenges

- Operations on multi-dimensional arrays
- Conditional logic with masks
- Reshaping and combining arrays with unequal dimensions
- Dealing with NaN in complex datasets
- Eigenvalue decomposition
- Matrix algebra for AI applications

---

## What’s Next?

Now that I’ve mastered NumPy, I’m moving on to:
-  [Pandas](https://pandas.pydata.org/): for structured data analysis
-  Data Visualization with Matplotlib/Seaborn
-  Machine Learning foundations using scikit-learn

---

## Author

Made with love by **Nauraa**  — self-taught AI engineer in progress 

---
