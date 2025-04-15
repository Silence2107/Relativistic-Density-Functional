# ARTICLE : https://arxiv.org/pdf/2204.03611

import numpy as np
import argparse

import sys

import rdf_module as rdf

parser = argparse.ArgumentParser(description='2 flavour RDF for neutral matter in beta equilibrium',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# subgroups
required = parser.add_argument_group('Required arguments')
required.add_argument('--eta_v', type=float, required=True,
                      help='Dimensionless vector coupling')
required.add_argument('--eta_d', type=float, required=True,
                      help='Dimensionless diquark coupling')

constants = parser.add_argument_group('Constants')
constants.add_argument('--f_pi', type=float, default=90.0,
                       help='Pion decay constant in MeV')
constants.add_argument('--m_pi', type=float,
                       default=140.0, help='Pion mass in MeV')
constants.add_argument('--m_sigma', type=float,
                       default=980.0, help='Sigma meson mass in MeV')
constants.add_argument('--m_e', type=float,
                       default=0.511, help='Electron mass in MeV')
constants.add_argument('--gluon_renorm_mass', type=float,
                       default=600.0, help='Gluon renormalized mass in MeV')
constants.add_argument('--q_cond', type=float, default=-
                       267.0, help='Vacuum quark condensate per flavour in MeV')

optimization = parser.add_argument_group('Optimization')
optimization.add_argument('--rel_tol', type=float, default=1e-10,
                          help='Relative tolerance for the optimizer')
optimization.add_argument('--guess_moment_scale', type=float,
                          default=500, help='Guess for the moment scale in MeV')
optimization.add_argument('--guess_effective_mass', type=float,
                          default=1000, help='Guess for the vacuum effective mass in MeV')
optimization.add_argument('--guess_gap', type=float, default=10,
                          help='Initial guess for the SC gap in SC phase in MeV')
optimization.add_argument('--guess_charge_chemical_potential', type=float,
                          default=-100, help='Initial guess for the charge chemical potential in MeV')
optimization.add_argument('--start', type=float,
                          default=223, help='mu_b* / 3 to start calculations with in MeV')
optimization.add_argument('--finish', type=float,
                          default=520, help='mu_b* / 3 to finish calculations with in MeV')
optimization.add_argument('--num_points', type=int,
                          default=100, help='Number of EoS points, linear in mu_b* / 3')

interface = parser.add_argument_group('Interface')
interface.add_argument('--non_verbose', action='store_true',
                       help='Suppress verbose output', default=False)
interface.add_argument('--output', type=str, default='',
                       help='Output file name')

args = parser.parse_args()


def main():
    mev3_fm3 = rdf.mev3_fm3
    # Actual vacuum quark condensate (MeV^3)
    q_bar_q_vac = 2 * args.q_cond ** 3
    # Bare quark mass from Gell-Mann-Oakes-Renner relation (MeV) for ud quarks
    bare_mass = -args.m_pi ** 2 * args.f_pi ** 2 / q_bar_q_vac
    vacuum_properties = rdf.instantiate_vacuum(
        args.guess_effective_mass, args.guess_moment_scale, bare_mass, q_bar_q_vac, args.m_pi, args.m_sigma, args.f_pi, args.eta_v, args.eta_d, args.rel_tol)
    if not args.non_verbose:
        print('Vacuum properties (MeV powers): ')
        for fields in vacuum_properties.__dataclass_fields__:
            print(f'{fields}: {getattr(vacuum_properties, fields)}')

    # Coupling renormalization scale (eq. 23-24 from https://www.mdpi.com/2571-712X/5/4/38)
    coupling_renorm_scale = 2 / np.pi ** 2 * \
        (9 / 8) ** (3 / 2) * args.gluon_renorm_mass ** 3

    # agree on the order
    u_index, d_index = 0, 1

    # Optimization loop
    mu_b_ef_selection = 3 * \
        np.linspace(args.start, args.finish, args.num_points)
    outf = open(args.output, 'w') if args.output != '' else sys.stdout
    guess_gap = args.guess_gap
    quess_q_bar_q = 0.2 * q_bar_q_vac
    guess_mu_q = args.guess_charge_chemical_potential
    confirmed_SC = False
    for mu_b_ef in mu_b_ef_selection:
        # in terms of optimized mu_q, we must require mu_q = -mu_e for beta equilibrium
        # and n_e = 2/3 * n_u - 1/3 * n_d for charge neutrality
        def neutr_eq(mu_q):
            # mu_q coincides with mu_q effective
            nonlocal guess_gap, quess_q_bar_q, confirmed_SC
            mu_ef_d = mu_b_ef / 3 - mu_q / 3
            mu_ef_u = mu_b_ef / 3 + 2 * mu_q / 3
            eff_chempots = np.zeros((2))
            eff_chempots[u_index] = mu_ef_u
            eff_chempots[d_index] = mu_ef_d

            opt_vals, errs = rdf.solve_eos_equations(eff_chempots, vacuum_properties,
                                                     coupling_renorm_scale, bare_mass, guess_gap, quess_q_bar_q, args.rel_tol, confirmed_SC)
            gap, q_bar_q = opt_vals
            if np.abs(q_bar_q) < 0.8 * np.abs(q_bar_q_vac):
                quess_q_bar_q = q_bar_q
                confirmed_SC = True
            # faster optimization for later calls with better gap guess
            guess_gap = np.abs(gap)
            # print(f"Mu*", eff_chempots, ": Vals", [gap, np.sign(
            #     q_bar_q) * (np.abs(q_bar_q)) ** (1 / 3)], ", Error", errs)
            if not args.non_verbose and (np.isnan(opt_vals).sum() != 0 or np.all(np.abs(errs) > args.rel_tol)):
                print(f"Mu*", eff_chempots, ": Vals", [gap, np.sign(
                    q_bar_q) * (np.abs(q_bar_q)) ** (1 / 3)], ", Error", errs)

            energy_density, pressure, n_b, densities, chempots = rdf.resolve_eos(eff_chempots, q_bar_q, gap, vacuum_properties,
                                                                                 coupling_renorm_scale, bare_mass, args.rel_tol)
            # account for ideal electrons
            # neutrality
            n_e_target = 2 / 3 * \
                densities[u_index] - 1 / 3 * densities[d_index]
            # beta equilibrium condition
            k_f = (mu_q ** 2 - args.m_e ** 2) ** (1 / 2) if - \
                mu_q > args.m_e else 0
            n_e = k_f ** 3 / (3 * np.pi ** 2)
            # impact on thermodynamics
            if -mu_q > args.m_e:
                energy_density += 1.0 / (4 * np.pi**2) * (-mu_q * k_f * (mu_q**2 -
                                                                         args.m_e**2 / 2) - args.m_e**4 / 2 * np.log((-mu_q + k_f) / args.m_e))
                pressure += 1.0 / (12 * np.pi**2) * (-mu_q * k_f * (mu_q**2 - 5 *
                                                                    args.m_e**2 / 2) + 3 * args.m_e**4 / 2 * np.log((-mu_q + k_f) / args.m_e))

            equation = (n_e - n_e_target) / (0.16 / mev3_fm3)  # scaled to nsat
            return equation, [energy_density, pressure, n_b, [*densities, n_e], [*chempots, -mu_q], gap, q_bar_q]

        # optimize mu_q
        mu_q_opt = rdf.root(lambda x: neutr_eq(
            x)[0], guess_mu_q, tol=args.rel_tol)
        mu_q = mu_q_opt.x[0]
        if confirmed_SC:
            guess_mu_q = mu_q
        energy_density, pressure, n_b, densities, chempots, gap, q_bar_q = neutr_eq(mu_q)[
            1]
        if not args.non_verbose:
            print('Mu ', chempots, ' MeV, Error: ', mu_q_opt.fun, sep='')
        # turn to typical units
        print(
            f'{energy_density * mev3_fm3 : 20.10e} {pressure * mev3_fm3 : 20.10e} {n_b * mev3_fm3 : 20.10e} {densities[0] * mev3_fm3 : 20.10e} {densities[1] * mev3_fm3 : 20.10e} {densities[2] * mev3_fm3 : 20.10e} {chempots[0] : 20.10e} {chempots[1] : 20.10e} {chempots[2] : 20.10e} {gap : 20.10e} {np.sign(q_bar_q) * (np.abs(q_bar_q)) ** (1 / 3) : 20.10e}', file=outf, flush=True)


main()
