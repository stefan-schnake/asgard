#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "../tensors.hpp"
#include "pde_base.hpp"

namespace asgard
{
// ---------------------------------------------------------------------------
//
// the "continuity 3d" pde
//
// 3D test case using continuity equation, i.e.,
//
// df/dt + v.grad(f)==0 where v={1,1,1}, so
//
// df/dt = -df/dx -df/dy - df/dz
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_lenard_bernstein_3d : public PDE<P>
{
public:
  PDE_lenard_bernstein_3d(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields used to check correctness of specification
  static int constexpr num_dims_           = 3;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 9;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = false;

  static P constexpr nu = 1.0e3;
  static P constexpr u_x = 1.0;
  static P constexpr u_y = 1.0;
  static P constexpr u_z = 1.0;
  static P constexpr th = 1.0;

  //
  // function definitions needed to build up the "dimension", "term", and
  // "source" member objects below for this PDE
  //

  // specify initial condition vector functions...
  static fk::vector<P>
  initial_condition_dim0(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::exp(-x_v*x_v/(2.0*th)); });
    return fx;
  }
  static fk::vector<P>
  initial_condition_dim1(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::exp(-x_v*x_v/(2.0*th)); });
    return fx;
  }
  static fk::vector<P>
  initial_condition_dim2(fk::vector<P> const x, P const t = 0)
  {
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

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dt      = x_range / std::pow(2, dim.get_level());
    return dt;
  }

  // g-funcs for terms (optional)
  static P g_func_identity(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }
  static P g_func_nu(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return nu;
  }
  static P g_func_t1_d1(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -1.0;
  }
  static P g_func_t2_d2(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -1.0;
  }

  // define dimensions
  inline static dimension<P> const dim0_ =
      dimension<P>(-6.0,                   // domain min
                   6.0,                    // domain max
                   2,                      // levels
                   3,                      // degree
                   initial_condition_dim0, // initial condition
                   volume_jacobian_dV,
                   "x"); // name

  inline static dimension<P> const dim1_ =
      dimension<P>(-6.0,                   // domain min
                   6.0,                    // domain max
                   2,                      // levels
                   3,                      // degree
                   initial_condition_dim1, // initial condition
                   volume_jacobian_dV,
                   "y"); // name

  inline static dimension<P> const dim2_ =
      dimension<P>(-6.0,                   // domain min
                   6.0,                    // domain max
                   2,                      // levels
                   3,                      // degree
                   initial_condition_dim2, // initial condition
                   volume_jacobian_dV,
                   "z"); // name

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_, dim1_,
                                                               dim2_};

  // define terms

  // default mass matrix (only for lev_x=lev_y=etc)
  inline static partial_term<P> const partial_term_I_ =
      partial_term<P>(coefficient_type::mass, g_func_identity, g_func_identity,
                      flux_type::central, boundary_condition::periodic,
                      boundary_condition::periodic);
  inline static term<P> const I_ = term<P>(false,  // time-dependent
                                           "mass", // name
                                           {partial_term_I_});

  // sd mass_matrix (nu f,g)
  inline static partial_term<P> const partial_term_nu_ =
      partial_term<P>(coefficient_type::mass, g_func_nu, g_func_identity,
                      flux_type::central, boundary_condition::periodic,
                      boundary_condition::periodic);
  inline static term<P> const nu_ = term<P>(false,  // time-dependent
                                           "mass", // name
                                           {partial_term_nu_});

  // sd_term d_v(vf)
  static P g_func_v(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(time);
    return x;
  }
  inline static partial_term<P> const partial_term_dvvf_ = partial_term<P>(
      coefficient_type::div, g_func_v, g_func_identity, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous);
  inline static term<P> const term_dvvf_ = term<P>(false, // time-dependent
                                                    "d_v(vf)", // name
                                                    {partial_term_t0_d0});  

  /*  div(vf)  */                                         

  // term 0 -- d_x(xf)
  inline static std::vector<term<P>> const terms0_ = {term_dvvf_, I_, nu_};

  // term 1 -- d_y(yf)
  inline static std::vector<term<P>> const terms1_ = {nu_, term_dvvf_, I_};

  // term 2 -- d_z(zf)
  inline static std::vector<term<P>> const terms2_ = {I_, nu_, term_dvvf_};


  /*  div(-uf) where u=(u_x,u_y,u_z) */

  // sd term d_x(-u_xf)
  static P g_func_u_x(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -u_x;
  }
  inline static partial_term<P> const partial_term_u_x_ =
      partial_term<P>(coefficient_type::div, g_func_u_x, g_func_identity,
                      flux_type::central, 
                      boundary_condition::dirichlet, boundary_condition::dirichlet,
                      homogeneity::homogeneous, homogeneity::homogeneous);
  inline static term<P> const term_u_x_ = term<P>(false,  // time-dependent
                                           "u_x", // name
                                           {partial_term_u_x_});

  // term 3 -- d_x(-u_xf)
  inline static std::vector<term<P>> const terms3_ = {term_u_x_, I_, nu_};

  // sd term d_y(-u_yf)
  static P g_func_u_y(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -u_y;
  }
  inline static partial_term<P> const partial_term_u_y_ =
      partial_term<P>(coefficient_type::div, g_func_u_y, g_func_identity,
                      flux_type::central,
                      boundary_condition::dirichlet, boundary_condition::dirichlet,
                      homogeneity::homogeneous, homogeneity::homogeneous);
  inline static term<P> const term_u_y_ = term<P>(false,  // time-dependent
                                           "u_y", // name
                                           {partial_term_u_y_});

  // term 4 -- d_y(-u_yf)
  inline static std::vector<term<P>> const terms4_ = {nu_, term_u_y_, I_};

  // sd term d_z(-u_zf)
  static P g_func_u_z(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -u_z;
  }
  inline static partial_term<P> const partial_term_u_z_ =
      partial_term<P>(coefficient_type::div, g_func_u_z, g_func_identity,
                      flux_type::central, 
                      boundary_condition::dirichlet, boundary_condition::dirichlet,
                      homogeneity::homogeneous, homogeneity::homogeneous);
  inline static term<P> const term_u_z_ = term<P>(false,  // time-dependent
                                           "u_z", // name
                                           {partial_term_u_z_});

  // term 5 -- d_z(-u_zf)
  inline static std::vector<term<P>> const terms5_ = {I_, nu_, term_u_z_};

  /*  div(sqrt(th)q), q = sqrt(th)\grad f  */

  // sd term d_v(sqrt(th)q), q = sqrt(th)d_vf
  static P g_func_sqrt_th(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return std::sqrt(th);
  }
  inline static partial_term<P> const partial_term_div_th =
      partial_term<P>(coefficient_type::div, g_func_sqrt_th, g_func_identity,
                      flux_type::central, 
                      boundary_condition::dirichlet, boundary_condition::dirichlet,
                      homogeneity::homogeneous, homogeneity::homogeneous);
  inline static partial_term<P> const partial_term_grad_th =
      partial_term<P>(coefficient_type::grad, g_func_sqrt_th, g_func_identity,
                      flux_type::central, 
                      boundary_condition::dirichlet, boundary_condition::dirichlet,
                      homogeneity::homogeneous, homogeneity::homogeneous);
  inline static term<P> const term_diff_th_ = term<P>(false,  // time-dependent
                                           "grad", // name
                                           {partial_term_div_th,partial_term_grad_th});

  // term 6 -- d_xx(th f)
  inline static std::vector<term<P>> const terms6_ = {term_diff_th_, I_, nu_};

  // term 7 -- d_yy(th f)
  inline static std::vector<term<P>> const terms7_ = {nu_, term_diff_th_, I_};

  // term 8 -- d_zz(th f)
  inline static std::vector<term<P>> const terms8_ = {I_, nu_, term_diff_th_};


  // Collect terms
  inline static term_set<P> const terms_ = {terms0_, terms1_, terms2_,
                                            terms3_, terms4_, terms5_
                                            terms6_, terms7_, terms8_};
};
} // namespace asgard
