# $\color{lightblue}{\texttt{TNrNLA}}$<p align="center">


This repository contains **Tensor Network randomized Numerical Linear Algebra** ($\texttt{TNrNLA}$), a custom research library developed for the paper *Linear algebra at exponential scale
via tensor network dimension reduction* ([link](https://arxiv.org/abs/2606.15350)) by
<p align="center">
  <a href="https://chriscamano.github.io/">Chris Camaño</a>,
  <a href="https://www.ethanepperly.com/">Ethan N. Epperly</a>,
  <a href="https://ram900.com/">Raphael A. Meyer</a>,
  and <a href="https://tropp.caltech.edu/">Joel A. Tropp</a>.
</p>
<img width="2910" height="930" alt="image" src="https://github.com/user-attachments/assets/d84d89dd-46f9-48f4-a3fa-4286c159b846" />

$\texttt{TNrNLA}$ uses tensor networks to design new algorithms for large-scale numerical linear algebra, with optimized data structures and routines for matrix product states ($\mathrm{MPS}$), matrix product operators ($\mathrm{MPO}$), and MPS column matrices. It also includes randomized numerical linear algebra primitives such as sketching, low-rank approximation, and variance-reduced stochastic trace estimation, implemented directly in terms of tensor network operations.

$\texttt{TNrNLA}$ is designed to be familiar and easy to use. Most tensor network contractions are hidden behind a high level interface to allow practitioners to write code that mirrors traditional linear algebra syntax. In particular, an expression such as


```python
A @ x
```

may represent an *exponentially* large matrix-vector product computed using a randomized tensor network contraction algorithm such as Sucessive Randomized Compression [[16]](https://arxiv.org/abs/2504.06475). 

<div align="center">

<table>
  <tr>
    <td align="center">

<strong>See the end of this README to learn how to install 
$\texttt{TNrNLA}$ using a single Linux command.</strong>

  </tr>
</table>

</div>

Tutorials that demonstrate *machine-precision* linear algebra at exponential scale, along with step-by-step instructions for using $\texttt{TNrNLA}$ on other exponentially large problems, are available in the TNrNLA Tutorial directory at the root of this repository.

# ✦ $\color{lightblue}\textbf{ What this library contains}$ 

## 1. $\color{lightblue}\textbf{ Randomized numerical linear algebra}$
- **Randomized SVD** [[1]](https://arxiv.org/abs/0909.4061)
- **Nyström approximations**
  - Randomized Nyström decomposition [[2]](https://dl.acm.org/doi/10.5555/1046920.1194916) [[3]](https://arxiv.org/abs/1303.1849) with a judiciously [stabilized](https://epubs.siam.org/doi/10.1137/22M1538648) variant  
- **Variance-reduced trace estimation**
  - Hutch++ [[7]](https://arxiv.org/abs/2010.09649)  
  - Nyström++ [[8]](https://arxiv.org/abs/2109.10659)  
  - XTrace / XNysTrace with resphering [[9]](https://arxiv.org/abs/2301.07825)
- **Fast Structured sketching operators**
  - Khatri–Rao sketching operators and tensor network representations
    
## 2. $\color{lightblue}\textbf{  Efficient implementations of 1D tensor networks}$ 
- **$\mathrm{MPS}/\mathrm{MPS}$ algebra**
  - Compressed binary operations `+`, `-`, `*`, `/` and in-place `+=`, `-=`, `*=`, `/=`
  - $\mathrm{MPS}$ inner products and norms
  - $\mathrm{MPS}$ outer products 
  - $\mathrm{MPO}$ trace
- $\mathrm{MPS}/\mathrm{MPO}$ visualizers (three dimensional rendering etc) $$\color{green}\text{new!}$$
- **$\mathrm{MPS}/\mathrm{MPO}$ orthogonalization**
  - Left-to-right and right-to-left orthogonalization
  - Mixed orthogonal forms
- **Exact methods**
  - Exact $\mathrm{MPO}\mathrm{MPS}$ products and exact $\mathrm{MPS}/\mathrm{MPO}$ compression [[12]](https://epubs.siam.org/doi/10.1137/090752286)
- **Deterministic methods for compressed $\mathrm{MPO}\mathrm{MPS}$ products**
  - Zip-up [[13]](https://arxiv.org/abs/1002.1305)  
  - Fitting (variational) method [[14]](https://arxiv.org/abs/cond-mat/0407066)  
  - Density matrix method [[15]](https://quantum-journal.org/papers/q-2024-12-27-1580/)
  - 
## 3. $\color{lightblue}\textbf{ New (and existing) randomized tensor network methods and tools}$ 
- **Randomized compression and rounding**
  - Optimized Successive Randomized Compression (SRC) for the compressed $\mathrm{MPO}\mathrm{MPS}$ [[16]](https://arxiv.org/abs/2504.06475)  
  - Randomized \mathrm{MPS} rounding methods [[17]](https://arxiv.org/abs/2110.04393)
- Isotropically normalized Gaussian random matrix product states ($\mathrm{rMPSs}$) $$\color{green}\text{new-ish!}$$  
- **Optimized implementations of Tensorized random projections ($\mathrm{TRP}$)** [[18]](https://arxiv.org/abs/2003.05101) [[19]](https://arxiv.org/abs/2105.00105) 
  - $\mathrm{MPS} Column Matrix -\mathrm{MPO}$ product via SRC $$\color{green}$$  
  - $\mathrm{MPS} Column Matrix-\mathrm{MPS}$ products $$\color{green}$$  
  - $\mathrm{MPS} Column Matrix-\mathrm{TRP}$ products $$\color{green}$$  
  - $\mathrm{MPS} Column Matrix$ vector and matrix multiplication $$\color{green}$$  
- **$\mathrm{MPS}$ versions of rNLA algorithms**
  - Optimized MPS Gram matrix calculations $$\color{green}\text{new!}$$  
  - $\mathrm{MPS}$ Nyström approximation $$\color{green}\text{new!}$$
  - $\mathrm{MPS}$ Girard–Hutchinson with optimized quadratic form computation $$\color{green}\text{new!}$$
  - $\mathrm{MPS}$ Hutch++ / Nyström++ $$\color{green}\text{new!}$$
  - $\mathrm{MPS}$ XTrace / XNysTrace with resphering $$\color{green}\text{new!}$$
## 4. $\color{lightblue}\textbf{Tensor network algorithms for 1D quantum physics}$ 
  * **Ground state preparation**
    * DMRG-2 for ground state preparation
  * **Hamiltonian and operator constructions**
    * Finite state machine constructions for various local Hamiltonians
    * Reduced density MPOs from ground state MPSs
    * Sparse Hamiltonians
  * **Time evolution**
    * TEBD-2 for real and imaginary time evolution
    * TDVP-1 for real and imaginary time evolution
    * TDVP-1 with global Krylov subspace enrichment
  * **Thermal states and trace based estimation**
    * Gibbs state preparation via $\mathrm{MPS}$ Nyström plus $\mathrm{MPS}$ XNysTrace $$\color{green}\text{new!}$$
    * Partition function estimation via tensor network variance reduced trace estimation $$\color{green}\text{new!}$$
  * **Entanglement and higher trace moments**
    * Estimation of higher trace moments $\mathrm{Tr}(\rho_A^n)$ for Rényi entropies and related quantities $$\color{green}\text{new!}$$

---

# ✦ $\color{lightblue}\textbf{Gallery}$

---

# ✦ $\color{lightblue}\textbf{How to Install TNrNLA with a single command}$

$\texttt{TNrNLA}$ is designed to be fast out of the box. Since tensor network workloads rely heavily on large dense linear algebra and memory sensitive tensor contractions, we compile the core routines against the
[Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
whenever it is available. On many AMD and Intel CPUs this can substantially speed up operations such as `SVD`, `QR`, and large dense tensor contractions. Several performance critical paths are also implemented as custom `C` and `C++` kernels.

To create a python environment where you can start using $\texttt{TNrNLA}$ simply run the following command from the root of the repository.

```bash
chmod +x SETUP.sh
./SETUP.sh tnrnla-mkl
```
> **Note for macOS users**  
> If you run the installer on macOS, you will be prompted with
>
> ```text
> $ Are you on a Mac and want to skip MKL + incrementalqr C++ build? (y/n)
> ```
>
> If you answer `y` or `yes`, the installer will skip MKL and use [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) instead. It will also skip building the custom C++ `incrementalqr` extension (which depends on MKL) and fall back to the slightly slower `scipy` implementation effecting the runtime of randomized MPO-MPS products.

---



