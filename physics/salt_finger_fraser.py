"""
Salt-finger convection in a fully periodic domain, in the non-dimensionalisation of
Fraser, van Kan, Knobloch, Julien & Liu (2025, JFM 1020 R1, eqs 2.1-2.5), which follows
Xie, Julien & Knobloch (2017, section 3.1), with the reduced models of the paper's section 4.
Builds on the shared DoubleDiffusion base in double_diffusion_base.py.

Units:
    length       d = (kT nu / g aT bT)^(1/4)   finger width (as in SaltFinger with Ra = 1)
    time         d^2 / kS                       salinity diffusion time
    velocity     kS / d
    temperature  tau bT d
    salinity     bS d / R
Full equations (paper 2.2-2.5; momentum multiplied by Sc and temperature divided by tau, so
that every dt has unit coefficient; p is Sc times the paper's pressure):
    dt u + u.grad u = -grad p + Sc (T - S) ez + Sc lap u,      div u = 0
    dt T + u.grad T + w/tau = lap T / tau
    dt S + u.grad S + R w   = lap S,                            R = 1 + eps
Parameters (params dict):
    Sc  = nu/kS = Pr/tau                      Schmidt number (Pr may be given instead)
    tau = kS/kT                               inverse Lewis number
    eps = R - 1,  R = 1/(Rrho tau)            supercriticality (0 = onset in an unbounded domain)
All three can be changed in place with set_param (e.g. continuation in 'eps').

Models (params['model'], default 'full'; paper section 4):
    'full'   the equations above, (2.2)-(2.5)
    'LPN'    low-Peclet-number limit: diagnostic temperature  lap T = w  (tau -> 0),
             full momentum equation
    'IFSC'   inertia-free salt convection (Xie et al. 2017; paper 4.1-4.2):
             lap T = w  and  0 = -grad p + Sc (T - S) ez + Sc lap u    (tau -> 0, Sc -> infinity)
    'MIFSC'  modified IFSC (paper 4.14-4.15): as IFSC, but the horizontal mean flow U(z, t)
             (the k_perp = 0 part of u) keeps its inertia and Reynolds stress,
                 dt U + dz <u' w'>_perp = Sc dz^2 U,
             while the fluctuations u' = u - U obey the inertia-free balance. Keeps the
             (helical) mean flow that IFSC loses.
S obeys (2.5) in full in every model. The reduced models are differential-algebraic (no dt on
T, nor on u or u'); tau then only enters the diagnostics. Units are those of the full model.

Relation to SaltFinger (Ra = 1, Pr = Sc tau, Rrho = 1/((1 + eps) tau)):
    t = tau t_SF,  u = u_SF/tau,  T = T_SF/tau,  S = (1 + eps) S_SF,  periods Tp = tau Tp_SF,
    <w S> = (1 + eps)/tau <w S>_SF,  <w T> = <w T>_SF/tau^2,  growth rates s = s_SF/tau.
    SaltFinger options: stokes + temperature_feedback = 'IFSC', temperature_feedback = 'LPN'.
"""
import dedalus.public as de
import numpy as np
from .double_diffusion_base import DoubleDiffusion

MODELS = ('full', 'LPN', 'IFSC', 'MIFSC')


class SaltFingerFraser(DoubleDiffusion):
    def __init__(self, params, sizes, bounds, bounded=False, mode='ecs', dealias=3/2, comm=None):
        params = dict(params)
        if 'Sc' not in params:
            if 'Pr' not in params:
                raise KeyError("SaltFingerFraser needs 'Sc' (or 'Pr') in params")
            params['Sc'] = params['Pr']/params['tau']
        params.pop('Pr', None)                       # Sc is the only viscosity parameter
        for key in ('tau', 'eps'):
            if key not in params:
                raise KeyError(f"SaltFingerFraser needs '{key}' in params")
        if 'stokes' in params or 'temperature_feedback' in params:
            raise KeyError("SaltFingerFraser: set params['model'] to 'LPN', 'IFSC' or 'MIFSC' "
                           "instead of 'stokes' / 'temperature_feedback'")
        names = {m.lower(): m for m in MODELS}
        model = str(params.get('model', 'full')).lower()
        if model not in names:
            raise ValueError(f"params['model'] = {params['model']!r}; choose one of {MODELS}")
        params['model'] = names[model]
        super().__init__(params, sizes, bounds, bounded, mode, dealias, comm)

    @property
    def model_name(self):
        """'full', 'LPN', 'IFSC' or 'MIFSC'."""
        return self.params['model']

    def _hmean(self, A):
        """Horizontal average: the k_perp = 0 part of A. Written with Integrate, because
        Dedalus (3.0) cannot take dt of an Average (no new_operand) but can of an Integrate."""
        if self.dim == 2:
            return de.Integrate(A, 'x')/self.bounds[0]
        return de.Integrate(A, ('x', 'y'))/(self.bounds[0]*self.bounds[1])

    def _equations(self, evp):
        """Equation strings of the chosen model. IVP: dt(q), nonlinear terms on the right.
        EVP (linearised about u_eq, te_eq, sa_eq, in a frame moving with c): dt(q) ->
        sigma*q - c@grad(q), and u@grad(q) -> u@grad(q_eq) + u_eq@grad(q) on the left."""
        model = self.model_name
        if evp:
            dt = lambda q: f"sigma*{q} - c@grad({q})"
            adv = lambda q: f" + u@grad({q}_eq) + u_eq@grad({q}) = 0"
            adv_mean = " + hmean(u@grad(u_eq) + u_eq@grad(u)) = 0"
        else:
            dt = lambda q: f"dt({q})"
            adv = lambda q: f" = - u@grad({q})"
            adv_mean = " = - hmean(u@grad(u))"
        forcing = "grad(p) - Sc*lap(u) - Sc*(te - sa)*ez + tau_u"
        eqs = ["trace(grad(u)) + tau_p = 0", "integ(p) = 0"]
        if model in ('full', 'LPN'):
            eqs.append(f"{dt('u')} + {forcing}" + adv('u'))
        elif model == 'IFSC':                        # (4.2): inertia-free momentum
            eqs.append(f"{forcing} = 0")
        else:                                        # MIFSC (4.14)-(4.15): only the mean flow has inertia
            eqs.append(f"{dt('hmean(u)')} + {forcing}" + adv_mean)
        if model == 'full':
            eqs.append(f"{dt('te')} - lap(te)/tau + w/tau + tau_te" + adv('te'))
        else:                                        # (4.1): diagnostic temperature
            eqs.append("- lap(te) + w + tau_te = 0")
        eqs.append(f"{dt('sa')} - lap(sa) + (1 + eps)*w + tau_sa" + adv('sa'))
        eqs += [f"integ({v}) = 0" for v in ('u', 'te', 'sa')]
        return eqs

    def build_ivp_problem(self):
        tau_p = self.dist.Field(name='tau_p')
        tau_u = self.dist.VectorField(self.coords, name='tau_u')
        tau_te = self.dist.Field(name='tau_te')
        tau_sa = self.dist.Field(name='tau_sa')
        ez = self.coords.unit_vector_fields(self.dist)[-1]

        # PARAMETERS (updated in place by set_param through the <name>_param fields)
        self.Sc_param = self.dist.Field(name='Sc_param'); self.Sc_param['g'] = self.params['Sc']
        self.tau_param = self.dist.Field(name='tau_param'); self.tau_param['g'] = self.params['tau']
        self.eps_param = self.dist.Field(name='eps_param'); self.eps_param['g'] = self.params['eps']

        ns = {'np': np, 'Sc': self.Sc_param, 'tau': self.tau_param, 'eps': self.eps_param,
              'ez': ez, 'w': self.u @ ez, 'hmean': self._hmean}
        vars = [self.p, self.u, self.te, self.sa,
                tau_p, tau_u, tau_te, tau_sa]
        self.ivp_problem = de.IVP(vars, namespace=ns)
        for eq in self._equations(evp=False):
            self.ivp_problem.add_equation(eq)

    def build_evp_problem(self, coupled=True):
        """Linearisation of the IVP about the base state (u_eq, te_eq, sa_eq), in a frame
        moving with velocity c (traveling waves), for the same model as the IVP. Eigenvalues
        are growth rates in units of kS/d^2. coupled: base fields on all bases; False: z-only
        base fields. Used by FluidModel.solve_evp; needs build_ivp_problem first."""
        sigma = self.dist.Field(name='sigma')
        tau_p = self.dist.Field(name='tau_p_evp')
        tau_u = self.dist.VectorField(self.coords, name='tau_u_evp')
        tau_te = self.dist.Field(name='tau_te_evp')
        tau_sa = self.dist.Field(name='tau_sa_evp')
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        u_eq, te_eq, sa_eq = self.evp_base_fields(coupled)

        ns = {'Sc': self.Sc_param, 'tau': self.tau_param, 'eps': self.eps_param,
              'ez': ez, 'w': self.u @ ez, 'c': self.evp_frame_velocity(), 'hmean': self._hmean,
              'u_eq': u_eq, 'te_eq': te_eq, 'sa_eq': sa_eq,
              'sigma': sigma, 'tau_p': tau_p, 'tau_u': tau_u, 'tau_te': tau_te, 'tau_sa': tau_sa}
        variables = [self.p, self.u, self.te, self.sa, tau_p, tau_u, tau_te, tau_sa]
        evp = de.EVP(variables, eigenvalue=sigma, namespace=ns)
        for eq in self._equations(evp=True):
            evp.add_equation(eq)
        return evp

    def _diagnostics(self):
        """Volume averages. Ft = <w T>, Fs = <w S> (the paper's fluxes). Nu = 1 - tau^2 <w T>
        and Sh = 1 - <w S>/R are the same numbers as in SaltFinger. Kinetic energy budget
        d<|u|^2/2>/dt = I - D, with I = Sc <(T - S) w> = I_t + I_s and D = Sc <|grad u|^2>."""
        ez = self.coords.unit_vector_fields(self.dist)[-1]
        w = self.u @ ez
        Sc, tau, R = self.params['Sc'], self.params['tau'], 1 + self.params['eps']
        grad_u = de.grad(self.u)
        grad_u2 = sum(de.dot(grad_u @ e, grad_u @ e) for e in self.coords.unit_vector_fields(self.dist)) # |grad u|^2, all components
        return {'Ft': de.Average(w*self.te),
                'Fs': de.Average(w*self.sa),
                'Nu': 1 - tau**2*de.Average(w*self.te),          # Nusselt number
                'Sh': 1 - de.Average(w*self.sa)/R,               # Sherwood number
                'D': de.Average(Sc*grad_u2),                     # viscous dissipation
                'I': de.Average(Sc*(self.te - self.sa)*w),       # kinetic energy input by buoyancy
                'I_t': de.Average(Sc*self.te*w),                 # ... by temperature-induced buoyancy
                'I_s': de.Average(-Sc*self.sa*w)}                # ... by salinity-induced buoyancy

    def get_flow_properties(self):
        vals = {k: op.evaluate()['g'].real for k, op in self._diagnostics().items()}
        if self.dist.comm.rank == 0:
            return {k: float(v.item()) for k, v in vals.items()}
        else:
            return {}

    def set_snapshots(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        snapshots = solver.evaluator.add_file_handler(self.odir+'snapshots', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        snapshots.add_task(self.te, name='te')
        snapshots.add_task(self.sa, name='sa')
        snapshots.add_task(self.u, name='u')

    def set_flowproperties(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        flowproperties = solver.evaluator.add_file_handler(self.odir+'flowproperties', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        for name, op in self._diagnostics().items():
            flowproperties.add_task(op, name=name)

    def set_meanprofiles(self, solver, sim_dt=1.0, max_writes=1000, mode='overwrite'):
        if self.dim == 2:
            h_mean = lambda A: de.Average(A, 'x')
        else:
            h_mean = lambda A: de.Average(A, ('x', 'y'))
        meanprofiles = solver.evaluator.add_file_handler(self.odir+'meanprofiles', sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        meanprofiles.add_task(h_mean(self.u), name='u')    # horizontally averaged velocity
        meanprofiles.add_task(h_mean(self.te), name='te')  # horizontally averaged temperature
        meanprofiles.add_task(h_mean(self.sa), name='sa')  # horizontally averaged salinity

