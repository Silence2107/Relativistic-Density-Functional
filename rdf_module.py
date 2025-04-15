# ARTICLE : https://arxiv.org/pdf/2204.03611

import numpy as np
from scipy.optimize import root
from scipy.integrate import quad

from dataclasses import dataclass

@dataclass
class VacuumProperties:
    instantiated: bool
    vacuum_error: float
    g_ps_vac: float
    g_s_vac: float
    g_v_vac: float
    g_d_vac: float
    alpha: float
    d_coupling: float
    q_bar_q_vac: float
    effective_mass_vac: float
    moment_scale: float
    vacuum_pressure: float

mev3_fm3 = 1 / 197.327 ** 3

def integrate(func, a, b, **kwargs):
    return quad(func, a, b, full_output=1, **kwargs)

def coupling_running(arg, renorm_scale, vacuum_properties : VacuumProperties, rel_tol, channel):
    # V,D couplings running (eq. 23-24 from https://www.mdpi.com/2571-712X/5/4/38)
    if channel == 'v':
        g_v0 = vacuum_properties.g_v_vac
        return g_v0 / (1 + (np.abs(arg) / renorm_scale) ** (2 / 3))
    elif channel == 'd':
        g_d0 = vacuum_properties.g_d_vac

        def eq(g_d):
            return g_d / g_d0 - 1 / (1 + (np.abs(arg / (2 * g_d)) / renorm_scale) ** (2 / 3))
        gd = root(eq, g_d0, tol=rel_tol)
        return gd.x[0]
    # PS coupling running (dictated by eq. 5)
    elif channel == 'ps':
        a = vacuum_properties.alpha
        q_bar_q_vac = vacuum_properties.q_bar_q_vac
        d0 = vacuum_properties.d_coupling
        arg_mf = (1 + a) * q_bar_q_vac ** 2 - arg ** 2
        if arg_mf <= 0:
            return 0
        return d0 / 3 * arg_mf ** (-2 / 3)
    else:
        print('Error: unknown coupling channel renormalization')
        exit(1)


def propagator_integral(eff_chempots, effective_mass, moment_scale, p_sqr):
    # calculate the propagator integral (eq. 27)
    if p_sqr >= 4 * effective_mass ** 2:
        print(
            'Error: p ** 2 >= 4 * effective_mass ** 2, propagator_integral is ill defined')
        exit(1)
    # formfactor part (flavour independent)

    def integrand1(k):
        eff_energy = np.sqrt(k ** 2 + effective_mass ** 2)
        return k ** 2 / (2 * np.pi ** 2) * np.exp(-k ** 2 / moment_scale ** 2) / (eff_energy * (4 * eff_energy ** 2 - p_sqr))
    res = 12 * integrate(integrand1, 0, np.inf)[0]
    # FD part (flavour sum)
    for eff_chempot in eff_chempots:
        # theta function
        top_momentum_sqr = eff_chempot ** 2 - effective_mass ** 2
        if top_momentum_sqr < 0:
            continue
        k_limit = np.sqrt(top_momentum_sqr)

        def integrand2(k):
            eff_energy = np.sqrt(k ** 2 + effective_mass ** 2)
            return -k ** 2 / (2 * np.pi ** 2) / (eff_energy * (4 * eff_energy ** 2 - p_sqr))
        res += 6 * integrate(integrand2, 0, k_limit)[0]
    return res


def instantiate_vacuum(effective_mass_guess, moment_scale_guess, bare_mass, q_bar_q_vac, m_pi, m_sigma, f_pi, eta_v, eta_d, rel_tol):
    # find vacuum parameters from known constants

    def score_minimize(x):
        effective_mass = x[0]
        moment_scale = x[1]
        # vacuum condensate calculation (eq. 22 in vacuum)

        def eq1():
            # integrate vacuum condensate equation (normalized)
            def integrand(k):
                return k ** 2 / (2 * np.pi ** 2) * np.exp(-k ** 2 / moment_scale ** 2) * effective_mass / np.sqrt(k ** 2 + effective_mass ** 2)
            integral = integrate(integrand, 0, np.inf)[0]
            return 1 + 12 * integral / q_bar_q_vac

        # pion pole (eq. 31)
        def eq2():
            # this equation is joint with GMOR (normalized)
            return (1 - (effective_mass - bare_mass) * effective_mass *
                    propagator_integral([], effective_mass, moment_scale, m_pi ** 2) / f_pi ** 2)

        return eq1(), eq2()

    # optimize the vacuum couplings
    x0 = [effective_mass_guess, moment_scale_guess]
    # optimize eqs
    # def aggregate_score(x):
    #    return sum(score_minimize(x))
    res = root(score_minimize, x0, tol=rel_tol)

    # vacuum couplings
    # eq.3 + effective mass definition
    g_ps_vac = 0.5 * (bare_mass - res.x[0]) / q_bar_q_vac
    g_s_vac = 0.5 / ((m_sigma ** 2 - 4 * res.x[0] ** 2) *
                     propagator_integral([], res.x[0], res.x[1], m_sigma ** 2) - q_bar_q_vac / res.x[0])  # eq. 29
    g_v_vac = g_s_vac * eta_v
    g_d_vac = g_s_vac * eta_d
    # field potential parameters
    # eq. 34 (it has typos) + eq. 33 (it has typos too); the expression here is corrected!
    alpha = - (4 / 3) / (1 + 2 * g_s_vac *
                         q_bar_q_vac / (res.x[0] - bare_mass))
    # eq. 33 (it has typos); the expression here is corrected!
    d_coupling = -(bare_mass - res.x[0]) / (2 / 3 *
                                            (-q_bar_q_vac) ** (-1 / 3) * alpha ** (-2 / 3))

    # vacuum pressure (eq. 16 in vacuum)
    def vacuum_quark_pressure_integrand(k):
        return 12 * k ** 2 / (2 * np.pi ** 2) * np.exp(-k ** 2 / res.x[1] ** 2) * np.sqrt(k ** 2 + res.x[0] ** 2)
    vacuum_quark_pressure = integrate(
        vacuum_quark_pressure_integrand, 0, np.inf)[0]
    vacuum_pressure = vacuum_quark_pressure - d_coupling * \
        (-q_bar_q_vac) ** (2 / 3) * (alpha ** (1 / 3) + 2 / 3 * alpha ** (-2 / 3))
    # return ({'g_ps_vac': g_ps_vac, 'g_s_vac': g_s_vac, 'g_v_vac': g_v_vac, 'g_d_vac': g_d_vac,
    #          'alpha': alpha, 'd_coupling': d_coupling, 'q_bar_q_vac': q_bar_q_vac,
    #          'effective_mass_vac': res.x[0], 'moment_scale': res.x[1], 'vacuum_pressure': vacuum_pressure},
    #         {'vacuum_error': score_minimize(res.x)}, res.success)
    return VacuumProperties(res.success, res.fun ,g_ps_vac, g_s_vac, g_v_vac, g_d_vac, alpha, d_coupling, q_bar_q_vac,
                            res.x[0], res.x[1], vacuum_pressure)


def solve_eos_equations(eff_chempots, vacuum_properties : VacuumProperties, coupling_renorm_scale, bare_mass, guess_gap, guess_q_bar_q, rel_tol, confirmed_SC=False):
    # resolve gap and quark condensate from self-consistency equations

    # we now optimize gap, q_bar_q to match thermodynamic consistency (eq. 21-22)
    def score_minimize(x):
        gap = x[0]
        q_bar_q = x[1]

        moment_scale = vacuum_properties.moment_scale
        q_bar_q_vac = vacuum_properties.q_bar_q_vac

        g_d = coupling_running(
            gap, coupling_renorm_scale, vacuum_properties, rel_tol, 'd')
        g_ps = coupling_running(
            q_bar_q, coupling_renorm_scale, vacuum_properties, rel_tol, 'ps')

        # dictated by eq. 3
        eff_mass = bare_mass - 2 * g_ps * q_bar_q
        # eq. 13 w/ omega interpretation
        # eff_chempots = [ch - 2 * g_v * q_sqr for ch in chempots]

        def eff_energy(k):
            return np.sqrt(k ** 2 + eff_mass ** 2)

        # gap self-consistency (zero gap is excluded) (eq. 21)
        def eq1():
            res = -1  # 0 = RHS, NORMALIZED to gap
            for eff_chempot in eff_chempots:
                # part w/ formfactor (both antiparticles and particles)
                def integrand(k):
                    calc = 0
                    for a in [1, -1]:
                        energy_w_chempot = eff_energy(k) - a * eff_chempot
                        # gap it
                        dispersion_relation = np.sqrt(
                            energy_w_chempot ** 2 + gap ** 2)
                        if energy_w_chempot < 0:
                            dispersion_relation = -dispersion_relation
                        calc += 1 / dispersion_relation
                    return 4 * g_d * k ** 2 / (2 * np.pi ** 2) * np.exp(- (k / moment_scale) ** 2) * calc
                res += integrate(integrand, 0, np.inf)[0]
                # part w/ FD (particles only)
                top_momentum_sqr = eff_chempot ** 2 - eff_mass ** 2
                if top_momentum_sqr > 0:
                    k_fermi = np.sqrt(top_momentum_sqr)

                    def integrand(k):
                        energy_w_chempot = eff_energy(k) - eff_chempot
                        rel_energy_w_chempot = np.sign(
                            energy_w_chempot) * np.sqrt(energy_w_chempot ** 2 + gap ** 2)
                        return -8 * g_d * k ** 2 / (2 * np.pi ** 2) / rel_energy_w_chempot
                    res += integrate(integrand, 0, k_fermi)[0]
            return res

        # quark condensate self-consistency (eq. 22)
        def eq2():
            # similar to eq. 20, so we will use the same structure
            res = -q_bar_q  # 0 = RHS equation
            for eff_chempot in eff_chempots:
                top_momentum_sqr = eff_chempot ** 2 - eff_mass ** 2

                # blue quark contribution
                # FD portion (particles only)
                if top_momentum_sqr > 0:
                    k_fermi = np.sqrt(top_momentum_sqr)

                    def integrand(k):
                        return k ** 2 / (2 * np.pi ** 2) * eff_mass / eff_energy(k)
                    res += 2 * integrate(integrand, 0, k_fermi)[0]
                # Formfactor portion (particles and antiparticles)

                def integrand(k):
                    return k ** 2 / (2 * np.pi ** 2) * eff_mass / eff_energy(k) * np.exp(-(k / moment_scale)**2)
                res -= 2 * integrate(integrand, 0, np.inf)[0]

                # red, green contribute equally
                # antiparticle contribution
                def integrand(k):
                    energy_w_chempot = eff_energy(k) + eff_chempot
                    return (k ** 2 / (2 * np.pi ** 2) * np.exp(- (k / moment_scale) ** 2) *
                            (energy_w_chempot ** 2 / (gap ** 2 + energy_w_chempot ** 2)) ** (1 / 2)) * eff_mass / eff_energy(k)
                res -= 2 * integrate(integrand, 0, np.inf)[0]

                # particle contribution, under Fermi surface
                # then contribution is akin to antiparticle one
                def integrand(k):
                    energy_w_chempot = eff_energy(k) - eff_chempot
                    return (k ** 2 / (2 * np.pi ** 2) * np.exp(- (k / moment_scale) ** 2) *
                            (energy_w_chempot ** 2 / (gap ** 2 + energy_w_chempot ** 2)) ** (1 / 2)) * eff_mass / eff_energy(k)
                res -= 2 * integrate(integrand, 0, np.inf)[0]

                # particle contribution, above Fermi surface
                if top_momentum_sqr > 0:
                    k_fermi = np.sqrt(top_momentum_sqr)
                    # portion w/ delta function
                    res -= 2 / np.pi ** 2 * gap * eff_mass * k_fermi * \
                        (np.exp(-k_fermi ** 2 / moment_scale ** 2) - 1)
                    # antiparticle-like part already incorporated above
                    # part w/ FD

                    def integrand(k):
                        energy_w_chempot = eff_energy(k) - eff_chempot
                        return -(k ** 2 / (2 * np.pi ** 2) * eff_mass / eff_energy(k) *
                                 (energy_w_chempot ** 2 / (gap ** 2 + energy_w_chempot ** 2)) ** (1 / 2))
                    res -= 4 * integrate(integrand, 0, k_fermi)[0]
            return res / (-q_bar_q_vac)
        return np.array([eq1(), eq2()])

    # root, function residue, termination message
    best_selection = [np.zeros((2)), np.ones((2)), "Unoptimized"]

    condensate_guesses = [guess_q_bar_q, vacuum_properties.q_bar_q_vac]
    if confirmed_SC:
        condensate_guesses.pop()
    # optimize gap and q_bar_q
    for gap_opt in guess_gap * np.array([1]):
        for q_bar_q_opt in condensate_guesses:
            x0 = [gap_opt, q_bar_q_opt]
            res = root(score_minimize, x0, tol=rel_tol)
            g_ps = coupling_running(
                res.x[1], coupling_renorm_scale, vacuum_properties, rel_tol, 'ps')
            eff_mass = bare_mass - 2 * g_ps * res.x[1]
            if sum(res.fun ** 2) < sum(best_selection[1] ** 2) and np.isnan(res.x).sum() == 0 and eff_mass > 0:
                best_selection = [res.x, res.fun, res.message]
                if np.all(np.abs(best_selection[1]) < rel_tol):
                    break
    return best_selection


def resolve_eos(eff_chempots, q_bar_q, gap, vacuum_properties : VacuumProperties, coupling_renorm_scale, bare_mass, rel_tol):
    # calculate nb, P, eps, nu, nd (prescription after eq. 20-22)

    moment_scale = vacuum_properties.moment_scale
    g_d = coupling_running(gap, coupling_renorm_scale, vacuum_properties, rel_tol, 'd')
    g_ps = coupling_running(
        q_bar_q, coupling_renorm_scale, vacuum_properties, rel_tol, 'ps')

    # calculate effective mass
    eff_mass = bare_mass - 2 * g_ps * q_bar_q

    def eff_energy(k):
        return np.sqrt(k ** 2 + eff_mass ** 2)

    # 1. calculate densities

    densities = []
    # density calculation from omega meson self-consistency (eq. 20)
    for eff_chempot in eff_chempots:
        n_f = 0
        # blue quark contribution is analytic (Fermi-like)
        top_momentum_sqr = eff_chempot ** 2 - eff_mass ** 2
        if top_momentum_sqr > 0:
            n_f += top_momentum_sqr ** (3 / 2) / \
                (3 * np.pi ** 2)

        # red, green contribute equally
        # antiparticle contribution
        def integrand(k):
            energy_w_chempot = eff_energy(k) + eff_chempot
            return (k ** 2 / (2 * np.pi ** 2) * np.exp(- (k / moment_scale) ** 2) *
                    (energy_w_chempot ** 2 / (gap ** 2 + energy_w_chempot ** 2)) ** (1 / 2))
        n_f += 2 * integrate(integrand, 0, np.inf)[0]

        # particle contribution, under Fermi surface
        # then contribution is akin to antiparticle one
        def integrand(k):
            energy_w_chempot = eff_energy(k) - eff_chempot
            return (k ** 2 / (2 * np.pi ** 2) * np.exp(- (k / moment_scale) ** 2) *
                    (energy_w_chempot ** 2 / (gap ** 2 + energy_w_chempot ** 2)) ** (1 / 2))
        n_f -= 2 * integrate(integrand, 0, np.inf)[0]

        # particle contribution, above Fermi surface
        if top_momentum_sqr > 0:
            k_fermi = np.sqrt(top_momentum_sqr)
            # portion w/ delta function
            n_f -= 2 / np.pi ** 2 * gap * eff_chempot * k_fermi * \
                (np.exp(-k_fermi ** 2 / moment_scale ** 2) - 1)
            # antiparticle-like part already incorporated above
            # part w/ FD

            def integrand(k):
                energy_w_chempot = eff_energy(k) - eff_chempot
                return -(k ** 2 / (2 * np.pi ** 2) *
                         (energy_w_chempot ** 2 / (gap ** 2 + energy_w_chempot ** 2)) ** (1 / 2))
            n_f -= 4 * integrate(integrand, 0, k_fermi)[0]
        densities.append(n_f)
    q_sqr = sum(densities)

    g_v = coupling_running(q_sqr, coupling_renorm_scale,
                           vacuum_properties, rel_tol, 'v')

    # 2. calculate pressure

    # vacuum level
    pressure = -vacuum_properties.vacuum_pressure

    # ideal gas contribution from quarks
    for eff_chempot in eff_chempots:
        k_f_sqr = eff_chempot ** 2 - eff_mass ** 2
        if k_f_sqr > 0:
            k_f = np.sqrt(k_f_sqr)
            mu = eff_chempot
            m = eff_mass
            pressure += 1.0 / (12 * np.pi**2) * (mu * k_f * (mu**2 - 5 *
                                                             m**2 / 2) + 3 * m**4 / 2 * np.log((mu + k_f) / m))

    # quark pressure
    def quark_pressure_integrand(k):
        energy_tot = 0
        # summarize energies over all flavours, colours, types
        for eff_chempot in eff_chempots:
            for a in [1, -1]:
                energy_w_chempot = eff_energy(k) - a * eff_chempot
                # gap it
                dispersion_relation = np.sqrt(energy_w_chempot ** 2 + gap ** 2)
                if energy_w_chempot < 0:
                    dispersion_relation = -dispersion_relation
                # blue
                energy_tot += energy_w_chempot
                # red, green
                energy_tot += 2 * dispersion_relation
        return k ** 2 / (2 * np.pi ** 2) * np.exp(-k ** 2 / moment_scale ** 2) * energy_tot

    # this integral must be calculated as precisely to avoid roundoff
    if max(eff_chempots) > eff_mass:
        # there will be discontinuities
        limits = [0, np.inf]
        for eff_chempot in eff_chempots:
            top_momentum_sqr = eff_chempot ** 2 - eff_mass ** 2
            if top_momentum_sqr > 0:
                # k_fermi corresponds to the discontinuity
                k_fermi = np.sqrt(top_momentum_sqr)
                limits.append(k_fermi)
        # sort
        limits.sort()
        # integrate over all intervals
        for i in range(len(limits) - 1):
            pressure += integrate(quark_pressure_integrand,
                                  limits[i], limits[i + 1], epsrel=1e-15)[0]
    else:
        pressure += integrate(quark_pressure_integrand,
                              0, np.inf, epsrel=1e-15)[0]

    # contribution generated by discontinuity of dispersion relation
    for eff_chempot in eff_chempots:
        top_momentum_sqr = eff_chempot ** 2 - eff_mass ** 2
        if top_momentum_sqr > 0:
            k_fermi = np.sqrt(top_momentum_sqr)
            pressure += 2 * k_fermi ** 3 * gap / (3 * np.pi ** 2)

    # mean field potential, might be incurred from g_ps (follows from eq.5)
    pressure -= vacuum_properties.d_coupling ** (1.5) / \
        np.sqrt(3 * g_ps)

    # self-energy contribution
    pressure -= 2 * g_ps * q_bar_q ** 2

    # vector repulsion
    omega = -2 * g_v * q_sqr
    pressure += (omega / 2) ** 2 / g_v

    # diquark attraction
    pressure -= (gap / 2) ** 2 / g_d

    # V, D rearrangement terms (eq. 17 from https://www.mdpi.com/2571-712X/5/4/38)
    arg = q_sqr / coupling_renorm_scale
    pressure += coupling_renorm_scale ** 2 * (arg ** 2 * g_v - 3 / 2 * vacuum_properties.g_v_vac * (
        arg ** (4 / 3) - 2 * arg ** (2 / 3) + 2 * np.log(1 + arg ** (2 / 3))
    ))
    arg = np.abs(gap) / (2 * g_d * coupling_renorm_scale)
    pressure -= coupling_renorm_scale ** 2 * (arg ** 2 * g_d - 3 / 2 * vacuum_properties.g_d_vac * (
        arg ** (4 / 3) - 2 * arg ** (2 / 3) + 2 * np.log(1 + arg ** (2 / 3))
    ))

    # 3. Energy density
    chempots = [eff_chempot - omega for eff_chempot in eff_chempots]
    energy_density = -pressure
    for chempot, n_f in zip(chempots, densities):
        energy_density += chempot * n_f

    # eps, P, nb, [nf per muf*], [muf per muf*]
    return energy_density, pressure, q_sqr / 3, densities, chempots
