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
// Pitch angle scattering collision operator with parallel velocity Vlasov
//
// 1x and 2v.  velocity: r - radial, z = cos(theta) pitch angle
//
// Parallel velocity is given by v_|| = rz
//
// PDE: df/dt + rz.grad(f) = C(f)
//
// where C is the collision operator using diffusion in pitch angle. 
//
// Diffusion tensor A is given by
//
// C(f) = div(A\grad f) where A\hat{r} = 0\hat{r} and A\hat{z} = \hat{z}
//
// grad_v(f) = df/dr \hat{r} + \sqrt{1-z^2}/r df/dz \hat{z}
//
// div( V_r \hat{r} + V_z \hat{z} ) = 1/r^2 d/dr ( r^2 V_r ) + 
//                                      + 1/r d/dz ( sqrt{1-z^2} V_z )
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_pitch_angle_1x_2v : public PDE<P>
{
public:
  PDE_pitch_angle_1x_2v(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields used to check correctness of specification
  static int constexpr num_dims_           = 3;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 1;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;

  //
  // function definitions needed to build up the "dimension", "term", and
  // "source" member objects below for this PDE
  //

  // specify initial condition vector functions...
  static fk::vector<P>
  initial_condition_dim0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    for (int i=0;i < x.size();++i) {
      //fx[i] = 0.5*std::sin(PI * x[i]) + 1.0;
      fx[i]=1.0;
    }
    return fx;
  }
  static fk::vector<P>
  initial_condition_dim1(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    for (int i=0;i < x.size();++i) {
      //fx[i] = 1.0/std::pow(std::sqrt(2.0*PI*1.0),3.0)*std::exp(-x[i]*x[i]/(2.0*1.0));
      fx[i] = 1.0;
    }
    return fx;
  }
  static fk::vector<P>
  initial_condition_dim2(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    for (int i=0;i < x.size();++i) {
      fx[i] = 1.0;
    }
    return fx;
  }

  // Specify exact solution vectors/time function as initial condition

  static fk::vector<P> exact_solution_dim0(fk::vector<P> const x, P const t = 0)
  {
    return initial_condition_dim0(x,t);
  }
  static fk::vector<P> exact_solution_dim1(fk::vector<P> const x, P const t = 0)
  {
    return initial_condition_dim1(x,t);
  }
  static fk::vector<P> exact_solution_dim2(fk::vector<P> const x, P const t = 0)
  {
    return initial_condition_dim2(x,t);
  }

  // Volume jacobians

  static P dim0_dV(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }
  static P dim1_dV(P const x, P const time)
  {
    ignore(time);
    return x*x;
  }
  static P dim2_dV(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  static P exact_time(P const time) { 
    ignore(time);
    return 1.0; 
    }


  // CFL condition

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dt      = x_range / std::pow(2, dim.get_level());
    return dt;
  }

  // g-funcs for terms (optional)

  // g(x) = 1
  static P g_func_identity(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }
  // g(x) = -1
  static P g_func_neg_1(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }
  // g(x) = x
  static P g_func_x(P const x, P const time)
  {
    ignore(time);
    return x;
  }
  // g(x) = relu(x) = x if x > 0 else 0
  static P g_func_x_pos(P const x, P const time)
  {
    ignore(time);
    return x > 0 ? x : 0.0;
  }
  // g(x) = x if x < 0 else 0
  static P g_func_x_neg(P const x, P const time)
  {
    ignore(time);
    return x < 0 ? x : 0.0;
  }
  // g(x) = sqrt(1-x^2)
  static P g_func_sqrt_1mx2(P const x, P const time)
  {
    ignore(time);
    return std::sqrt(1.0-x*x);
  }

  // define dimensions
  inline static dimension<P> const dim0_ =
      dimension<P>(-1.0,                   // domain min
                   1.0,                    // domain max
                   4,                      // levels
                   3,                      // degree
                   initial_condition_dim0, // initial condition
                   dim0_dV,
                   "x"); // name

  inline static dimension<P> const dim1_ =
      dimension<P>(0,                      // domain min
                   10,                     // domain max
                   4,                      // levels
                   3,                      // degree
                   initial_condition_dim1, // initial condition
                   dim1_dV,
                   "r"); // name

  inline static dimension<P> const dim2_ =
      dimension<P>(-1.0,                   // domain min
                   1.0,                    // domain max
                   4,                      // levels
                   3,                      // degree
                   initial_condition_dim2, // initial condition
                   dim2_dV,
                   "z"); // name

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_, dim1_,
                                                               dim2_};
  //
  // Define terms
  //

  // Two terms for div_x(rzf) that need to be split because z changes sign on [-1,1]

  // First term for z>0.  |z|=z and we use upwinding

  // upwind div in x
  inline static partial_term<P> const partial_term_x_1_1 =
      partial_term<P>(
        coefficient_type::div, g_func_neg_1,
        partial_term<P>::null_gfunc, flux_type::central, 
        boundary_condition::periodic, boundary_condition::periodic,
        homogeneity::homogeneous, homogeneity::homogeneous,
                      {}, partial_term<P>::null_scalar_func,
                      {}, partial_term<P>::null_scalar_func,
                      dim0_dV);

  inline static term<P> const term1_x = term<P>(false,  // time-dependent
                                           "dx_downwind", // name
                                           {partial_term_x_1_1});

  // mass in r
  inline static partial_term<P> const partial_term_r_1_1 = partial_term<P>(
      coefficient_type::mass, g_func_identity, 
      partial_term<P>::null_gfunc, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic,
      homogeneity::homogeneous, homogeneity::homogeneous,
                      {}, partial_term<P>::null_scalar_func,
                      {}, partial_term<P>::null_scalar_func,
                      dim1_dV);

  inline static term<P> const term1_r = term<P>(false, // time-dependent
                                                    "mass_r", // name
                                                    {partial_term_r_1_1});

  // mass in z with g(x) = z.*(z > 0)
  inline static partial_term<P> const partial_term_z_1_1 = partial_term<P>(
      coefficient_type::mass, g_func_identity, 
      partial_term<P>::null_gfunc, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic,
      homogeneity::homogeneous, homogeneity::homogeneous,
                      {}, partial_term<P>::null_scalar_func,
                      {}, partial_term<P>::null_scalar_func,
                      dim2_dV);

  inline static term<P> const term1_z = term<P>(false, // time-dependent
                                                    "mass_z*(z>0)", // name
                                                    {partial_term_z_1_1});

  inline static std::vector<term<P>> const terms1_ = {term1_x, term1_r, term1_z};


  // Next term: if z < 0, then |z| = -z so we downwind

  // downwind div in x
  inline static partial_term<P> const partial_term_x_2_1 =
      partial_term<P>(coefficient_type::div, g_func_neg_1, dim0_dV,
                      flux_type::upwind, boundary_condition::periodic,
                      boundary_condition::periodic);

  inline static term<P> const term2_x = term<P>(false,  // time-dependent
                                           "dx_upwind", // name
                                           {partial_term_x_2_1});

  // mass in r not changed from previous term
  inline static term<P> const term2_r = term1_r;

  // mass in z with g(z) = z.*(z < 0)
  inline static partial_term<P> const partial_term_z_2_1 = partial_term<P>(
      coefficient_type::mass, g_func_x_neg, dim2_dV, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term2_z = term<P>(false, // time-dependent
                                                    "mass_z*(z<0)", // name
                                                    {partial_term_z_2_1});

  inline static std::vector<term<P>> const terms2_ = {term2_x, term2_r, term2_z};


  // Third term is diffusion in z in the pitch angle coordinates

  // mass in x
  inline static partial_term<P> const partial_term_x_3_1 = partial_term<P>(
      coefficient_type::mass, g_func_identity, dim0_dV, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term3_x = term<P>(false, // time-dependent
                                                    "mass_x", // name
                                                    {partial_term_x_3_1});  

  // mass in r (differs from above because of surface jacobian)
  inline static partial_term<P> const partial_term_r_3_1 = partial_term<P>(
      coefficient_type::mass, g_func_identity, g_func_x, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term3_r = term<P>(false, // time-dependent
                                                    "mass_r", // name
                                                    {partial_term_r_3_1});

  // LDG in z
  inline static partial_term<P> const partial_term_z_3_1 =
      partial_term<P>(coefficient_type::div, g_func_identity, g_func_sqrt_1mx2,
                      flux_type::downwind, boundary_condition::dirichlet,
                      boundary_condition::dirichlet);

   inline static partial_term<P> const partial_term_z_3_2 =
      partial_term<P>(coefficient_type::grad, g_func_identity, g_func_sqrt_1mx2,
                      flux_type::upwind, boundary_condition::neumann,
                      boundary_condition::neumann);

  inline static term<P> const term3_z = term<P>(false,  // time-dependent
                                           "LDG_z", // name
                                           {partial_term_z_3_1,partial_term_z_3_2});

  inline static std::vector<term<P>> const terms3_ = {term3_x, term3_r, term3_z};

  //inline static term_set<P> const terms_ = {terms1_, terms2_, terms3_};
  inline static term_set<P> const terms_ = {terms1_};

  inline static std::vector<source<P>> const sources_ = {};

  // define exact soln
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution_dim0, exact_solution_dim1, exact_solution_dim2};

  inline static scalar_func<P> const exact_scalar_func_ = exact_time;
};
} // namespace asgard
