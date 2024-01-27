# Optimization Methods Repository

This repository contains implementations and explanations of various optimization methods commonly used in mathematical optimization and machine learning. The implemented methods include Gradient Descent, Newton's Method, Lagrangian Optimization, Bisection, and more.

## Table of Contents

- [Introduction](#introduction)
- [Optimization Methods](#optimization-methods)
  - [Bisection Method](Bisection.py)
  - [Gradient Descent](Gradient.py)
  - [Newton raphson's Method](Newton_raph.py)
  - [Lagrangian Optimization](Lagrangian.py)
  - [Golden Section Method](Golden_Section.py)
  - [Quasi-Newton Method](Quasi_Newton.py)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Optimization methods play a crucial role in various fields, including machine learning, mathematical modeling, and decision-making processes. This repository serves as a collection of implementations and explanations for several optimization algorithms.

## Optimization Methods

### Bisection Method

The Bisection Method is a simple numerical technique for finding the roots of a real-valued function within a given interval.

### Gradient Descent

The Gradient Descent method is an iterative optimization algorithm for finding the minimum of a function. It is widely used in machine learning for training models.

### Newton's Method

Newton's Method is an iterative root-finding algorithm that converges quickly to the roots of a real-valued function.

### Lagrangian Optimization

Lagrangian Optimization involves maximizing or minimizing a function subject to equality constraints. It is commonly used in constrained optimization problems.

### Golden Section Method

The Golden Section Method is a one-dimensional optimization algorithm used to find the minimum of a unimodal function within a specified interval. It is an iterative algorithm that efficiently narrows down the search interval by dividing it into two segments of the golden ratio.

### Quasi-Newton Method

The Quasi-Newton Method is an iterative optimization algorithm used to find the minimum of a multivariate function. It belongs to the family of quasi-Newton methods, which aim to approximate the inverse Hessian matrix (second-order derivative) without explicitly computing it.

## Usage

To use the implementations in this repository, follow the instructions in each method's respective script. Ensure you have the necessary dependencies installed.

```bash
# Example command for running Gradient Descent
python Gradient.py
