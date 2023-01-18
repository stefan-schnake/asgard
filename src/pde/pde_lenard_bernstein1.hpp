#pragma once
#include "pde_base.hpp"

namespace asgard
{
// Example PDE using the 1D Diffusion Equation. This example PDE is
// time dependent (although not all the terms are time dependent). This
// implies the need for an initial condition.
// PDE: df/dt = d^2 f/dx^2

template<typename P>
class PDE_lenard_bernstein_1d : public PDE<P>
{
public:
  PDE_lenard_bernstein_1d(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 3;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = false;

  static P constexpr nu = 1.0e3;
  static P constexpr  u = 1.0;
  static P constexpr th = 1.0;

  static fk::vector<P>
  initial_condition_dim0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::exp(-x_v*x_v/(2.0*th)); });
    return fx;
  }

  static P volume_jacobian_dV(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 =
      dimension<P>(-6, 6, 3, 2, initial_condition_dim0, volume_jacobian_dV, "x");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0};

  /* Define terms */

  //// TERM_0 = d_v(vf)

  // g0(v) = nu*v
  static P g0(P const x, P const time)
  {
    ignore(time);
    return nu*x;
  }

  inline static const partial_term<P> partial_term_0 =
      partial_term<P>(coefficient_type::div, g0,
                      partial_term<P>::null_gfunc, flux_type::central,
                      boundary_condition::dirichlet, boundary_condition::dirichlet,
                      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const term_0 =
      term<P>(true, // time-dependent
              "",   // name
              {partial_term_0});

  //// TERM_1 = d_v(-uf)

  // g0(v) = nu*v
  static P g1(P const x, P const time)
  {
    ignore(time);
    ignore(x)
    return -u;
  }

  inline static const partial_term<P> partial_term_1 =
      partial_term<P>(coefficient_type::div, g1,
                      partial_term<P>::null_gfunc, flux_type::central,
                      boundary_condition::dirichlet, boundary_condition::dirichlet,
                      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const term_1 =
      term<P>(true, // time-dependent
              "",   // name
              {partial_term_1});

  //// TERM_2 = d_v(sqrt(th)*q), q = sqrt(th)*d_v(f)

   static P g2(P const x, P const time)
  {
    ignore(time);
    ignore(x)
    return std::sqrt(th*nu);
  }

  inline static const partial_term<P> partial_term_2 =
      partial_term<P>(coefficient_type::div, g2,
                      partial_term<P>::null_gfunc, flux_type::central,
                      boundary_condition::dirichlet, boundary_condition::dirichlet,
                      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static const partial_term<P> partial_term_3 = partial_term<P>(
      coefficient_type::grad, g2,
      partial_term<P>::null_gfunc, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);

  inline static term<P> const term_2 =
      term<P>(true, // time-dependent
              "",   // name
              {partial_term_2, partial_term_3});

  inline static std::vector<term<P>> const terms_0 = {term_0};
  inline static std::vector<term<P>> const terms_1 = {term_1};
  inline static std::vector<term<P>> const terms_2 = {term_2};
  inline static term_set<P> const terms_ = {terms_0, terms_1, terms_2};

  static P get_dt_(dimension<P> const &dim)
  {
    /* (1/2^level)^2 = 1/4^level */
    /* return dx; this will be scaled by CFL from command line */
    return std::pow(0.25, dim.get_level());
  }
};
} // namespace asgard
