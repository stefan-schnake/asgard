#pragma once
#include "pde_base.hpp"

namespace asgard
{
// 4D test case using relaxation problem
//
//  df/dt == div_v( (v-u(x))f + T(x)\grad_v f)
//
//  where the domain is (x,v_x,v_y,v_z).  The moments of f are constant x.
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_sphericalLB : public PDE<P>
{
public:
  PDE_sphericalLB(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_, moments_,
               do_collision_operator_)
  {}

private:
  static int constexpr num_dims_          = 2;
  static int constexpr num_sources_       = 0;
  static int constexpr num_terms_         = 5;
  static bool constexpr do_poisson_solve_ = false;
  // disable implicit steps in IMEX
  static bool constexpr do_collision_operator_ = false;
  static bool constexpr has_analytic_soln_     = true;
  static int constexpr default_degree          = 3;

  static P constexpr nu  = 1e3;
  static P constexpr T   = 1.0;
  static P constexpr u_f = 1.0;

//   static fk::vector<P>
//   initial_condition_dim_x(fk::vector<P> const &x, P const t = 0)
//   {
//     ignore(t);

//     fk::vector<P> fx(x.size());
//     std::transform(
//         x.begin(), x.end(), fx.begin(), [coefficient, T](P const x_v) -> P {
//           return 1.0;
//         });
//     return fx;
//   }

  static fk::vector<P>
  initial_condition_dim_r(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    P const T_init = 1.0;
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * T_init);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient, T_init](P const x_v) -> P {
          return (x_v < 3.5) ? 1.0/( 2.0 * std::pow(3.5,3) / 3.0 ) : 0.0;
          //return 2 * PI * std::pow(coefficient,3) * std::exp(-(0.5 / T_init) * std::pow(x_v, 2));
          //ignore(x_v); return 1.0;
        });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_th(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
          //ignore(x_v); return 1.0;
          return (x_v < 3.0*PI/4.0) & (x_v > PI/2.0) ? 2.0*std::sqrt(2.0) : 0.0;
        });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_phi(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
          ignore(x_v);
          return 1.0;
        });
    return fx;
  }

  // --- Specify Jacobians ---

  // Volume jacobian            = r^2 sin(th) dr dth dphi
  // Surface jacobian (r const) = r^2 sin(th)    dth dphi
  //                 (th const) = r   sin(th) dr     dphi
  //                (phi const) = r           dr dth

  static P jac_r(P const x, P const time = 0)
  {
    ignore(time);
    return x;
  }

  static P jac_r_sq(P const x, P const time = 0)
  {
    ignore(time);
    return x*x;
  }

  static P jac_th( P const x, P const time = 0)
  {
    ignore(time);
    return std::sin(x);
  }


  /*
  inline static dimension<P> const dim_x =
      dimension<P>(-1.0, 1.0, 3, default_degree,
                   {initial_condition_dim_r},
                   nullptr, "v1");
  */

  inline static dimension<P> const dim_r =
      dimension<P>(0.0, 7.0, 3, default_degree,
                   initial_condition_dim_r,
                   jac_r_sq, "r");

  inline static dimension<P> const dim_th =
      dimension<P>(0.0, M_PI, 3, default_degree,
                   initial_condition_dim_th,
                   jac_th, "th");

/*
  inline static dimension<P> const dim_phi =
      dimension<P>(0.0, 2*M_PI, 3, default_degree,
                   initial_condition_dim_phi,
                   nullptr, "phi");
*/                   

  inline static std::vector<dimension<P>> const dimensions_ = {dim_r, dim_th}; //, dim_phi

  /* Define the moments */
  static fk::vector<P> moment_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> f(x.size());
    std::fill(f.begin(), f.end(), 1.0);
    return f;
  }

  static fk::vector<P> moment_y_sq(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> f(x.size());
    std::transform(
        x.begin(), x.end(), f.begin(), [](P const x_v) -> P {
          return x_v*x_v;
        });
    return f;
  }

  static fk::vector<P> moment_sin_y(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> f(x.size());
    std::transform(
        x.begin(), x.end(), f.begin(), [](P const x_v) -> P {
          return std::sin(x_v);
        });
    return f;
  }

  inline static moment<P> const moment0 = // (f,1)_v
      moment<P>(std::vector<md_func_type<P>>(
          {{moment_y_sq,moment_sin_y}}));

  inline static std::vector<moment<P>> const moments_ = {}; //No moments for now

  /* build the terms */

  // Mass terms (for each coordinate direction)

  // r
  inline static const partial_term<P> mass_r_pterm = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_r_sq);

  inline static term<P> const mass_r =
      term<P>(false, // time-dependent
              "I",   // name
              {mass_r_pterm}, imex_flag::imex_implicit);

  // th
  inline static const partial_term<P> mass_th_pterm = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_th);

  inline static term<P> const mass_th =
      term<P>(false, // time-dependent
              "I",   // name
              {mass_th_pterm}, imex_flag::imex_implicit);

/*
  // phi
  inline static const partial_term<P> mass_phi_pterm = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const mass_phi =
      term<P>(false, // time-dependent
              "I",   // name
              {mass_phi_pterm}, imex_flag::imex_implicit);
*/

  // Constant Identity term

  inline static const partial_term<P> I_pterm = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const I_im =
      term<P>(false, // time-dependent
              "I",   // name
              {I_pterm}, imex_flag::imex_implicit);

  // Implcit Term 1
  // grad_v \cdot (vf)
  // in sphirical coordintes this is only radially dependent
  // since v = r\hat{r}

  static P nu_r_func(P const r, P const time = 0)
  {
    ignore(time);
    return nu * r;
  }

  inline static const partial_term<P> nu_v_pterm = partial_term<P>(
      coefficient_type::div, nu_r_func, nullptr, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_r_sq);

  inline static term<P> const nu_v_term =
      term<P>(false,  // time-dependent
              "I1_v", // name
              {nu_v_pterm}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_div_v_r = {nu_v_term, mass_th}; // , mass_phi

  // Implict Term 2
  // \grad_v \cdot (-u_f\hat{z} f ) -- radial direction
  // \hat{z} = cos(th)\hat{r} - sin(th)\hat{th}
  // Thus we need to modify the mass term for th

  static P nu_func(P const r, P const time = 0)
  {
    ignore(time);
    ignore(r);
    return nu;
  }

  inline static const partial_term<P> div_u_r_pterm = partial_term<P>(
      coefficient_type::div, nu_func, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_r_sq);

  inline static term<P> const div_u_r_term =
      term<P>(false,  // time-dependent
              "I1_v", // name
              {div_u_r_pterm}, imex_flag::imex_implicit);   

  static P neg_u_cos_th(P const th, P const time = 0)
  {
    ignore(time);
    return - u_f * std::cos(th);
  } 

  inline static const partial_term<P> surf_u_th_pterm = partial_term<P>(
      coefficient_type::mass, neg_u_cos_th, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_th);

  inline static term<P> const surf_u_th_term =
      term<P>(false,  // time-dependent
              "I1_v", // name
              {surf_u_th_pterm}, imex_flag::imex_implicit); 

  inline static std::vector<term<P>> const terms_div_u_r = {div_u_r_term, surf_u_th_term}; // , mass_phi


  // Implict Term 3
  // \grad_v \cdot (-u_f\hat{z} f ) -- th direction
  // \hat{z} = cos(th)\hat{r} - sin(th)\hat{th}
  
  inline static const partial_term<P> surf_r_pterm = partial_term<P>(
    coefficient_type::mass, nullptr, nullptr, flux_type::central,
    boundary_condition::periodic, boundary_condition::periodic,
    homogeneity::homogeneous, homogeneity::homogeneous,
    {},nullptr,{},nullptr,jac_r);

  inline static term<P> const surf_r_term =
      term<P>(false, // time-dependent
              "I",   // name
              {surf_r_pterm}, imex_flag::imex_implicit);

  static P u_sin_th_nu(P const th, P const time = 0)
  {
    ignore(time);
    return nu * u_f * std::sin(th);
  } 

  inline static const partial_term<P> div_u_th_pterm = partial_term<P>(
      coefficient_type::div, u_sin_th_nu, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_th);    

  inline static term<P> const div_u_th_term =
      term<P>(false, // time-dependent
              "I",   // name
              {div_u_th_pterm}, imex_flag::imex_implicit);        

  inline static std::vector<term<P>> const terms_div_u_th = {surf_r_term, div_u_th_term}; // , mass_phi

  // Diffusion term
  // grad_v \cdot ( T \grad_v f ) -- radial component

  // Used in all 3 diffusion terms
  static P sqrt_nu_theta(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return std::sqrt(T * nu);
  }

  inline static const partial_term<P> div_r_pterm = partial_term<P>(
      coefficient_type::div, sqrt_nu_theta, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_r_sq);

  inline static const partial_term<P> grad_r_pterm = partial_term<P>(
      coefficient_type::grad, sqrt_nu_theta, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_r_sq);

  inline static term<P> const diff_r_chain =
      term<P>(false,  // time-dependent
              "I3_v", // name
              {div_r_pterm, grad_r_pterm}, imex_flag::imex_implicit);

  inline static term<P> const surf_th_chain =
      term<P>(false,  // time-dependent
              "I3_v", // name
              {mass_th_pterm, mass_th_pterm}, imex_flag::imex_implicit);

/*
  inline static term<P> const surf_phi_chain = //this is identity
      term<P>(false,  // time-dependent
              "I3_v", // name
              {mass_phi_pterm, mass_phi_pterm}, imex_flag::imex_implicit);
*/

  inline static std::vector<term<P>> const terms_diff_r = {
      diff_r_chain, surf_th_chain}; // , surf_phi_chain

  // // Penalty Term -- radial

  // static P nu_theta(P const x, P const time = 0)
  // {
  //   ignore(x);
  //   ignore(time);
  //   return T * nu;
  // }

  // inline static const partial_term<P> penalty_r_pterm = partial_term<P>(
  //     coefficient_type::penalty, nu_theta, nullptr, flux_type::downwind,
  //     boundary_condition::neumann, boundary_condition::neumann,
  //     homogeneity::homogeneous, homogeneity::homogeneous,
  //     {},nullptr,{},nullptr,jac_r_sq);

  // inline static term<P> const penalty_r =
  //     term<P>(false,  // time-dependent
  //             "I3_v", // name
  //             {penalty_r_pterm}, imex_flag::imex_implicit);

  // inline static std::vector<term<P>> const terms_penalty_r = {
  //     penalty_r, mass_th}; //, mass_phi

  // Diffusion term
  // grad_v \cdot ( T \grad_v f ) -- azimuthal portion

  inline static term<P> const surf_r_chain =
      term<P>(false, // time-dependent
              "I",   // name
              {surf_r_pterm,surf_r_pterm}, imex_flag::imex_implicit);

  inline static const partial_term<P> div_th_pterm = partial_term<P>(
      coefficient_type::div, sqrt_nu_theta, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_th);

  inline static const partial_term<P> grad_th_pterm = partial_term<P>(
      coefficient_type::grad, sqrt_nu_theta, nullptr, flux_type::central,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::homogeneous, homogeneity::homogeneous,
      {},nullptr,{},nullptr,jac_th);

  inline static term<P> const diff_th_chain =
      term<P>(false,  // time-dependent
              "I3_v", // name
              {div_th_pterm, grad_th_pterm}, imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_diff_th = {
      surf_r_chain, diff_th_chain }; // , surf_phi_chain

  // Diffusion Term
  // grad_v \cdot ( T \grad_v f ) -- polar portion

  // inline static const partial_term<P> div_phi_pterm = partial_term<P>(
  //     coefficient_type::div, sqrt_nu_theta, nullptr, flux_type::central,
  //     boundary_condition::periodic, boundary_condition::periodic);

  // inline static const partial_term<P> grad_phi_pterm = partial_term<P>(
  //     coefficient_type::grad, sqrt_nu_theta, nullptr, flux_type::central,
  //     boundary_condition::periodic, boundary_condition::periodic);

  // inline static term<P> const diff_phi_chain =
  //     term<P>(false,  // time-dependent
  //             "I3_v", // name
  //             {div_phi_pterm, grad_phi_pterm}, imex_flag::imex_implicit);

  // inline static std::vector<term<P>> const terms_diff_phi = {
  //     surf_r_chain, surf_phi_chain, diff_phi_chain}; // second term is identity 
  //                                                    // due to surface jacobian

  inline static term_set<P> const terms_ = {terms_div_v_r,terms_div_u_r,terms_div_u_th,
                                            terms_diff_r,terms_diff_th};


  // Analytic solution -- Maxwellian at (0,0)
  static fk::vector<P>
  exact_dim_r(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    P const coefficient = 1.0 / std::sqrt(2.0 * PI * T);

    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [coefficient](P const x_v) -> P {
          return 2 * PI * std::pow(coefficient,3) * std::exp(-(0.5 / T) * std::pow(x_v, 2));
          //ignore(x_v); return 1.0;
        });
    return fx;
  }

  static fk::vector<P> exact_dim_th(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
          ignore(x_v);
          return 1.0;
        });
    return fx;
  }

/*
  static fk::vector<P> exact_dim_phi(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
          ignore(x_v);
          return 1.0;
        });
    return fx;
  }
*/

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_dim_r, exact_dim_th}; // , exact_dim_phi

  inline static scalar_func<P> const exact_scalar_func_ = {};

  static P get_dt_(dimension<P> const &dim)
  {
    ignore(dim);
    /* return dx; this will be scaled by CFL from command line */
    // return std::pow(0.25, dim.get_level());

    // TODO: these are constants since we want dt always based on dim 2,
    //  but there is no way to force a different dim for this function!
    // (Lmax - Lmin) / 2 ^ LevX * CFL
    return (6.0 - (-6.0)) / std::pow(2, 3);
  }

  /* problem contains no sources */
  inline static std::vector<source<P>> const sources_ = {};
};

} // namespace asgard
