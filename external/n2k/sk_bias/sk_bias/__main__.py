# See n2k/sk_bias/README.txt for a little documentation on how this code is used.

import sys
import argparse

from . import sk_bias


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

test = subparsers.add_parser('test')
check_interpolation = subparsers.add_parser('check_interpolation')

run_mcs = subparsers.add_parser('run_mcs')
run_mcs.add_argument('rms', type=float)
run_mcs.add_argument('n', type=int)
run_mcs.add_argument('--no-interp', action='store_true')
run_mcs.add_argument('mu_min', type=float, nargs='?')
run_mcs.add_argument('mu_max', type=float, nargs='?')

run_unquantized_mcs = subparsers.add_parser('run_unquantized_mcs')
run_unquantized_mcs.add_argument('n', type=int)

run_transit_mcs = subparsers.add_parser('run_transit_mcs')
run_transit_mcs.add_argument('nt', type=int)
run_transit_mcs.add_argument('ndish', type=int)
run_transit_mcs.add_argument('brightness', type=float)

make_plots = subparsers.add_parser('make_plots')
make_plots.add_argument('mu_min', type=float, nargs='?', default=2.0)
make_plots.add_argument('mu_max', type=float, nargs='?', default=50.0)

emit_code = subparsers.add_parser('emit_code')

args = parser.parse_args()

if args.command == 'test':
    sk_bias.test_ipow()
    sk_bias.test_fit_polynomial()
    sk_bias.test_mc_tracker()
elif args.command == 'check_interpolation':
    binterp = sk_bias.BiasInterpolator()
    sk_bias.check_bias_interpolation(binterp)
    sk_bias.check_sigma_interpolation(binterp)
elif args.command == 'run_mcs':
    pdf = sk_bias.Pdf(rms = args.rms)
    sk_bias.run_mcs(pdf, args.n, binterp = not args.no_interp, mu_min = args.mu_min, mu_max = args.mu_max)
elif args.command == 'run_unquantized_mcs':
    sk_bias.run_unquantized_mcs(args.n)
elif args.command == 'run_transit_mcs':
    sk_bias.run_transit_mcs(args.nt, args.ndish, args.brightness)
elif args.command == 'make_plots':
    sk_bias.make_fig1()
    sk_bias.make_fig2(mu_min=args.mu_min, mu_max=args.mu_max)
elif args.command == 'emit_code':
    sk_bias.emit_code()
else:
    parser.print_help()
    sys.exit(2)

