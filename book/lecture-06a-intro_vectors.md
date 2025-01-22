---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Chapter 6.1: A Deeper look at Vectors

## Learning Objectives

By the end of this lecture, you should be able to:

- Define a vector space and an inner product space.
- Understand basic vector operations, including addition, scalar multiplication, and the dot product.
- Define a matrix and matrix multiplication.


<!-- 1. **Encode Data in Vectors** by discretizing or featurizing data. -->
<!-- 2. **Apply Matrix Multiplications** for stoichiometric coefficients using Python and NumPy. -->
<!-- 3. **Apply the null space method** to find balanced coefficients for chemical reactions. -->
<!-- 4. **Interpret and generalize solutions** to balance hydrocarbon combustion reactions and other chemical equations. -->

## Why Vectors 
At its most immediate, encoding an object in a list of numbers is a natural thing to do.
* We might describe an object using multiple "features", each of which has a numeric value.  We might describe an apartment by its size in square feet, the number of bedrooms, which floor it is on.  Similarly, we might describe a student by their transcript: a number between 0 and 100 for every class they've taken.
* A convenient way to store a function on a computer is by storing its values at a list of points.
* To describe a molecule, we might write down its elemental composition: the number of carbons,  the number of hydrogens, the number of oxygens, and so forth.

However, by itself a  list of numbers is not very useful: to do science, we need ways to manipulate it.  To manipulate our lists, we first consider three main properties:
1. **Addition**: We can add two lists of numbers together by adding their corresponding entries.

$$ 
  \begin{bmatrix} 1 \\2 \\3 \end{bmatrix} + \begin{bmatrix} 4 \\5 \\6 \end{bmatrix} = \begin{bmatrix} 1+4 \\2+5 \\3+6 \end{bmatrix} = \begin{bmatrix} 5 \\7 \\9 \end{bmatrix} 
$$

2. **Scalar Multiplication**: We can multiply a list of numbers by a single number by multiplying each entry by that number.

$$ 
  2 \left( 
  \begin{bmatrix} 1 \\2 \\3 \end{bmatrix} \right) = 2 \left( \begin{bmatrix} 1 \\2 \\3 \end{bmatrix} \right) = \begin{bmatrix} 2 \cdot 1 \\2 \cdot 2 \\2 \cdot 3 \end{bmatrix} = \begin{bmatrix} 2 \\4 \\6 \end{bmatrix} 
$$

3. **Dot Product**: We can multiply two lists of numbers together by multiplying their corresponding entries and summing the result.

$$ \begin{bmatrix} 1 \\2 \\3 \end{bmatrix} \cdot \begin{bmatrix} 4 \\5 \\6 \end{bmatrix} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32 $$

We can visualize vectors as arrows in space.  The vector $\begin{bmatrix} 1 \\ 2 \end{bmatrix}$, for instance, is an arrow that starts at the origin and ends at the point $(1,2)$.

### A Formal Definition of a Vector
Formally, we have defined a set of minimal operations we expect our list to obey.  
We define a space of objects called **vectors** along with a second space of objects called **scalars**.
We define two operations: **vector addition** and **scalar multiplication**.

If, for any three vectors $u,v,w$ and two scalars $a,b$, we the following properties to hold,
1.  **Associativity**: $(u+v)+w = u+(v+w)$.
2.  **Commutativity**:  $u+v = v+u$.
3.  **Identity element for vector addition**: There is a vector $0$ such that $v+0 = v$ for all $v$.
4.  **Inverse**: For every vector $v$, there is a another vector $-v$ such that $v+(-v) = 0$.
5.  **Compatibility with Scalars**: $a(bv) = (ab)v$.
6.  **Distributivity of Scalars**: $a(u+v) = au + av$.
7.  **Distributivity of Vectors**: $(a+b)v = av + bv$.
8.  **Identity element for scalar multiplication**: $1v = v$.
then our vectors and scalars form a **vector space**.

We then further introduce a **dot product** operation, which takes two vectors and returns a scalar.  We require that the dot product be **bilinear**:
1.  **Linearity in the first argument**: $(au+v) \cdot w = a(u \cdot w) + v \cdot w$.
2.  **Symmetry**: $u \cdot v = v \cdot u$.
3.  **Positive Definite**: $u \cdot u \geq 0$ and $u \cdot u = 0$ if and only if $u = 0$.
Equipped with these operations, we have formed an **inner product space**.
(Note: you might have heard of the term "Hilbert Space" in your quantum mechanics classes.  A Hilbert space is an inner product space with a few more technical properties we won't discuss here.)

These somewhat abstract properties allow us to impose a consistent structure on a wide variety of objects.  For instance, we can define an inner product space of functions whose square-integral is finite.
Our "vectors" add elementwise:
$ (f+g)(x) = f(x) + g(x) $.  Similarly, a function times a scalar is defined by multiplying the function by the scalar:
$ (af)(x) = a f(x) $
For the dot product, we use the integral:
$ (f \cdot g) = \int f^*(x) g(x) dx $
where $f^*$ is the complex conjugate of $f$.
This structure shows up often in quantum mechanics, where vectors are wavefunctions.
However, this not the only inner product space we can define.  For instance, in statistical mechanics, we might define our vectors and scalars the same way, but instead define an inner product using the average against the Boltzmann density:
$ (f \cdot g) = \frac{1}{Z} \int f(x) g(x) E^{-\beta H(x)} $
where $\beta$ is the inverse temperature.

The key point between these examples is that we can define similar structures on a wide variety of objects.  This will be very useful computationally: we can approximate a function (e.g. a wavefunction) by a list of numbers but still expect much of the same mathematical structure to exist.

## Introduction to Matrices
We have a natural way of generalizing addition to vectors.  We now turn to a second question: how should we generalize multiplication?
In fact, we have already introduced one possible generalization: the dot product, which "multiplies" two vectors to produce a scalar.
This gives us an idea; if we have a vector with $M$ entries, we can perform $K$ different dot products to get a new vector with $K$ entries.

To help make this practical, we introduce a new notation.  We write the first vector in a dot product as a row of numbers (a "row vector") and the second vector as a column of numbers (a "column vector").

$$ \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32 $$

To take the dot product of a row vector and a column vector, we march along the row vector and down the column vector, multiplying the corresponding entries and summing the result as we go.

We can extend this idea to a matrix, which is a rectangular array of numbers.  Effectively, this is a set of row vectors stacked on top of each other.

$$ \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \begin{bmatrix} 7 \\ 8 \\ 9 \end{bmatrix} = \begin{bmatrix} 1 \cdot 7 + 2 \cdot 8 + 3 \cdot 9 \\ 4 \cdot 7 + 5 \cdot 8 + 6 \cdot 9 \end{bmatrix} = \begin{bmatrix} 50 \\ 122 \end{bmatrix} $$

As another example, if we had an arbitrary entries in a 2-by-2 matrix and a vector with two entries, we could write the matrix-vector product as below.

$$ \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} ax + by \\ cx + dy \end{bmatrix} $$

### Matrix Multiplication
Just as we stacked row vectors vertically, we can stack our column vectors horizontally.  This allows us to multiply two matrices together.
For instance, if we have two 2-by-2 matrices, we can multiply them together as follows.

$$ \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} w & x \\ y & z \end{bmatrix} = \begin{bmatrix} aw + by & ax + bz \\ cw + dy & cx + dz \end{bmatrix} $$

Observe that the $i,j$'th entry is the dot product of the $i$'th row of the first matrix and the $j$'th column of the second matrix.

Matrices don't have to be square or even have the same number of rows and columns.  However, to multiply two matrices together, the number of columns in the first matrix must equal the number of rows in the second matrix.
For instance, if we have a 2-by-3 matrix and a 3-by-2 matrix, we can multiply them together.

$$ \begin{bmatrix} a & b & c \\ d & e & f \end{bmatrix} \begin{bmatrix} w & x \\ y & z \\ u & v \end{bmatrix} = \begin{bmatrix} aw + bx + cu & ax + by + cv \\ dw + ex + fu & dx + ey + fv \end{bmatrix} $$

We could also multiply a 3-by-2 matrix by a 2-by-3 matrix.

$$ \begin{bmatrix} w & x \\ y & z \\ u & v \end{bmatrix} \begin{bmatrix} a & b & c \\ d & e & f \end{bmatrix} = \begin{bmatrix} wa + xb + uc & wb + xb + vc & wc + xb + vc \\ ya + zd + vf & yb + ze + vf & yc + zf + vf \end{bmatrix} $$

We could *also* multiply a 3-by-2 matrix by a 2-by-2 matrix.

$$ \begin{bmatrix} w & x \\ y & z \\ u & v \end{bmatrix} \begin{bmatrix} a & b \\ c & d \end{bmatrix} = \begin{bmatrix} wa + xb & wb + xd \\ ya + zc & yb + zd \\ ua + vb & ub + vd \end{bmatrix} $$

However, we could not multiply a 2-by-3 matrix by a 2-by-2 matrix.

$$ \begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix} \begin{bmatrix} w & x \\ y & z \end{bmatrix} = \text{ERROR!} $$

Note that matrix multiplication has some key differences from scalar multiplication.
1.  Matrix multiplication *is* associative: $(AB)C = A(BC)$.
2.  Matrix multiplication *is* distributive: $A(B+C) = AB + AC$.
3.  Matrix multiplication *is not* commutative: $AB = BA$.



### Basic Matrix Operations
Matrices inherent many of the properties of matrices:
1. **Addition**: We can add two matrices of the same size together by adding their corresponding entries.  For example,
$$ \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix} $$
2. **Scalar Multiplication**: We can multiply a matrix by a single number by multiplying each entry by that number.  For example,
$$ 2 \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 2 \cdot 1 & 2 \cdot 2 \\ 2 \cdot 3 & 2 \cdot 4 \end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix} $$
(In fact, the space of all matrices of a specific size is itself a vector space.)

In addition to these operations, we can also define a **transpose** operation.  The transpose of a matrix is the matrix with its rows and columns swapped.  For instance,
$$ \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix} $$
It should be easy to verify that the transpose operation satisfies the following properties:
1.  $(A^T)^T = A$.
2.  $(A + B)^T = A^T + B^T$.
3.  $(cA)^T = cA^T$.
4.  $(AB)^T = B^T A^T$.

### Some Examples of Matrices in Chemistry

#### First-order Rate Laws
Consider a collection of first-order rections,
\begin{align}
  A &\xrightarrow{k_{AB}} B \\
  A &\xrightarrow{k_{AC}} C \\
  B &\xrightarrow{k_{BC}} C \\
  C &\xrightarrow{k_{CA}} A
\end{align}





## The Identity and the Matrix Inversion
One of the most important matrices is the **identity matrix**, traditionally, denoted as $I$, which is a square matrix with ones on the diagonal and zeros elsewhere.  For instance, the 2-by-2 identity matrix is

$$ \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} $$

The identity matrix has the property that multiplying any matrix (or vector) by the identity matrix gives the original matrix: 
Just as we can define the inverse of a number, we can (attempt to) define the inverse of a matrix. 
The inverse of a matrix $A$ is a matrix $A^{-1}$ such that $AA^{-1} = A^{-1}A = I$.

$$\left( \begin{bmatrix} 2 & 3 & 1 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \right)^{-1} = \begin{bmatrix} -1/3 & -10 / 9 & 13 / 9 \\ 2/3 & 11 / 9 & 8 / 9 \\ -1/3 &  5 / 9 & -2 / 9 \end{bmatrix} $$

Not all matrices have inverses: any matrix who sends a vector to zero does not have an inverse.
To show this, let $v$ be a nonzero vector such that $Av = 0$.  If an inverse existed, we would have $v = I v = A^{-1} A v = A^{-1} 0 = 0$, which is a contradiction.

```{admonition} Exercise
:class: tip
Is the matrix 

$$ A =  \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} $$

invertible?  If so, find its inverse.  If not, find a vector $v$ such that $Av = 0$.
```