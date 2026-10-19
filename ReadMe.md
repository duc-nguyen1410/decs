# DECS (Dedelus-ECS): 

A **Dedalus**-based Solver finding **E**xact **C**oherent **S**tructures in Fluid Dynamics.

In DECS, we separately developed Python classes, including physical models `physics/` in fluid dynamics using [Dedalus](https://dedalus-project.org/), a solver finding ECS `ecs_core/` (equilibria, traveling wave, (relative) periodic orbits) as shown by examples bellow, and a continuation feature `continuation/`.

<table>
    <tr>
        <td align="center"><b>Equilibrium</b></td>
        <td align="center"><b>Traveling wave</b></td>
        <td align="center"><b>Periodic orbit</b></td>
        <td align="center"><b>Relative periodic orbit</b></td>
    </tr>
    <tr>
        <td><img src="examples/Lx0.6_WF/WF_Lx0.6.png" width="200"/></td>
        <td><img src="examples/Lx0.6_TM1/TM1.png" width="200"/></td>
        <td><img src="examples/Lx0.8_PO3/PO3.png" width="200"/></td>
        <td><img src="examples/Lx0.8_RPO2/RPO2.png" width="200"/></td>
    </tr>
</table>

Please read instructions for [conducting simulation](docs/simulation.md) of [available models](docs/models.md), [finding ECS](docs/find_ecs.md), and [parametric continuation](docs/continuation.md). Also, you can build [new equations](docs/build_equations.md) by yourself if needed.

<!-- Please cite as :  -->

Contact: PhD candicate Van Duc (Duc) Nguyen, School of Mechanical, Aerospace, and Manufacturing Engineering, University of Connecticut, CT, USA

Email: duc.nguyen@uconn.edu

This code is built upon Dedalus V3.0 from the https://dedalus-project.org/.