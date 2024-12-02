How to use the 'sk_bias' python module:

  1. Run unit tests:

      python -m sk_bias test

  2. (Subsection 1 and Fig 1 in notes) Study (b, sigma) versus (n, mu_true):

        # Do some MC runs to test bias/sigma calculation, and justify statement
	# in overleaf that finite-n only affects sigma by a few percent.
        python -m sk_bias run_mcs <rms> <n>

	# The 'make_plots' command will print out "Note: max Delta(b) = 0.01577..."
	# This justifies statement in notes that finite-n affects b by <~ 1.5%.
        python -m sk_bias make_plots   # fig1_left_panel.pdf, fig1_right_panel.pdf
     
  3. (Subsection 2.) Justify statement that bias and sigma interpolations are
     extremely accurate:

       python -m sk_bias check_interpolation

  4. (Subsection 3.) Explore residual SK-bias from finite-n and edge effects.

        python -m sk_bias make_plots   # fig1_left_panel.pdf, fig1_right_panel.pdf

  5. (Subsection 3.) Justify statement that finite-n/edge effects can also
     bias estimation of sigma by 10-20%.

        # Args are (rms, n, mu_min, mu_max).
	python -m sk_bias run_mcs 4.0 64 2 50
	python -m sk_bias run_mcs 5.0 64 2 50
