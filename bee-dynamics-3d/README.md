#  Program to Plot a Dynamical System (Lorenz Attractor)

##  Problem Statement
We are asked to **write a program to plot the following equations over time**.  
The equations define a **dynamical system** in which the position of the system changes with time.  
Think of this as a **bee moving in 3D space**. The goal is to simulate and plot the path taken by the bee.

The system is defined as:

\[
\dot{x} = a(y-x), \quad 
\dot{y} = bx - y - xz, \quad 
\dot{z} = xy - cz
\]

with parameters:

- \(a = 10\)  
- \(b = 28\)  
- \(c = 2.667\)  

Initial conditions:

\((x_0, y_0, z_0) = (0, 1, 1.05)\)

---

##  Flowchart

```text
          Start
            |
            V
     Set Parameters (a, b, c, dt)
            |
            V
   Initialize x0, y0, z0 values
            |
            V
   Repeat for N steps (simulation loop)
        ┌─────────────┐
        │ Compute dx/dt│
        │ Compute dy/dt│
        │ Compute dz/dt│
        └─────────────┘
            |
            V
    Update (x, y, z) using Runge-Kutta
            |
            V
    Store results in arrays
            |
            V
       Plot trajectory
            |
            V
          End
```
## Approach
Implemented the Lorenz system equations.

Used Runge-Kutta 4th order method for numerical integration (better accuracy than Euler).

Simulated the system for 10,000 steps.

Plotted:

3D trajectory (bee path in space)

2D projections: XY, XZ, YZ planes

Observed chaotic nature of the system → small changes in initial values diverge significantly.

## Results
3D Bee Path

XY Projection

XZ Projection

YZ Projection

## Conclusion
The Lorenz system shows chaotic dynamics.

The bee’s path is never repeating, forming the famous butterfly/figure-8 attractor.

The projections help visualize hidden structure in the chaos.

This simulation demonstrates how deterministic equations can still create unpredictable behavior.


<img width="567" height="581" alt="result" src="https://github.com/user-attachments/assets/705b9b28-2126-4ca9-a3df-ae171b31ac9d" />
