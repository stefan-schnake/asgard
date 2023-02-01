#pragma once
#include "pde_base.hpp"

namespace asgard
{
// 2D test case using continuity equation, i.e.,
//
//  df/dt == -v*\grad_x f + div_v( (v-u)f + theta\grad_v f) + source
//
//  where source is defined to give the analytic solution
//
//  f(x,v,t) = alpha(x,t)M(v) + beta(x,t)(1+v^2)M(v),
//      M(v) = 1/sqrt(pi)*exp(-(v-1)^2)
//
//  BC is peridoic in x
//  BC in v is all inflow in advection for v and Neumann for diffusion in v

template<typename P>
class PDE_vlasov_lb_source : public PDE<P>
{
public:
  PDE_vlasov_lb_source(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_, moments_)
  {
    param_manager.add_parameter(parameter<P>{"n", n});
    param_manager.add_parameter(parameter<P>{"u", u});
    param_manager.add_parameter(parameter<P>{"theta", theta});
  }

private:
  static int constexpr num_dims_           = 2;
  static int constexpr num_sources_        = 12;
  static int constexpr num_terms_          = 5;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;
  static int constexpr default_degree      = 3;

  static P constexpr nu = 0.0;

  static fk::vector<P>
  initial_condition_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return (std::abs(x_v) > 0.5) ? 1.0 : 0.0;
    });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_x_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(), [](P const x_v) -> P {
      return (std::abs(x_v) <= 0.5) ? 1.0 : 0.0;
    });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_0(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = 1.0 / std::sqrt(2.0 * PI);

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [coefficient](P const x_v) -> P {
                     return coefficient * std::exp(-std::pow(x_v, 2) / 2.0);
                   });
    return fx;
  }

  static fk::vector<P>
  initial_condition_dim_v_1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    P const coefficient = (1.0 / 8.0) / std::sqrt(2.0 * PI * (4.0 / 5.0));

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [coefficient](P const x_v) -> P {
                     return coefficient *
                            std::exp(-std::pow(x_v, 2) / (2.0 * (4.0 / 5.0)));
                   });
    return fx;
  }

  static P dV(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  /* Define analytic solution */
  static P
  alpha_(P const x, P const t = 0)
  {
    return (1.0+0.5*std::sin(2*M_PI*x))*(0.5*std::cos(2*t)+1.0);
  }

  static P dt_alpha_(P const x, P const t = 0)
  {
    return (1.0+0.5*std::sin(2*M_PI*x))*(-1.0)*std::sin(t);
  }

  static P dx_alpha_(P const x, P const t = 0)
  {
    return M_PI*std::cos(2*M_PI*x)*(0.5*std::cos(2*t)+1);
  }


  static P beta_(P const x, P const t = 0)
  {
    return (1.0+0.5*std::cos(2*M_PI*x))*(std::sin(t)+1);
  }
  static P dt_beta_(P const x, P const t = 0)
  {
    return (1.0+0.5*std::cos(2*M_PI*x))*(std::cos(t));
  }
  static P dx_beta_(P const x, P const t = 0)
  {
    return (-M_PI*std::sin(2*M_PI*x))*(std::sin(t)+1);
  }

  static fk::vector<P>
  soln_dim_x_0(fk::vector<P> const &x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = alpha_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P>
  soln_dim_x_1(fk::vector<P> const &x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = beta_(x[i],t);
    }
    return fx;
  }

  static P Mxwl_(P const x, P const t = 0)
  {
    ignore(t);
    return 1/std::sqrt(M_PI)*std::exp(-std::pow(x-1.0,2));

  }
  static P onepv2_(P const x, P const t = 0)
  {
    ignore(t);
    return 1.0+x*x;
  }

  static fk::vector<P>
  soln_dim_v_0(fk::vector<P> const &x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = Mxwl_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P>
  soln_dim_v_1(fk::vector<P> const &x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = onepv2_(x[i],t)*Mxwl_(x[i],t);
    }
    return fx;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 = dimension<P>(
      -1.0, 1.0, 4, default_degree,
      {soln_dim_x_0, soln_dim_x_1}, dV, "x");

  inline static dimension<P> const dim_1 = dimension<P>(
      -6.0, 6.0, 3, default_degree,
      {soln_dim_v_0, soln_dim_v_1}, dV, "v");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1};

  /* Define the moments */
  static fk::vector<P> moment0_f1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> f(x.size());
    std::fill(f.begin(), f.end(), 1.0);
    return f;
  }

  static fk::vector<P> moment1_f1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);
    return fk::vector<P>(x);
  }

  static fk::vector<P> moment2_f1(fk::vector<P> const &x, P const t = 0)
  {
    ignore(t);

    fk::vector<P> f(x.size());
    std::transform(x.begin(), x.end(), f.begin(),
                   [](P const &x_v) -> P { return std::pow(x_v, 2); });
    return f;
  }

  inline static moment<P> const moment0 = moment<P>(
      std::vector<md_func_type<P>>({{moment0_f1, moment0_f1, moment0_f1}}));
  inline static moment<P> const moment1 = moment<P>(
      std::vector<md_func_type<P>>({{moment0_f1, moment1_f1, moment0_f1}}));
  inline static moment<P> const moment2 = moment<P>(
      std::vector<md_func_type<P>>({{moment0_f1, moment2_f1, moment0_f1}}));

  inline static std::vector<moment<P>> const moments_ = {moment0, moment1,
                                                         moment2};

  /* Construct (n, u, theta) */
  static P n(P const &x, P const t = 0)
  {
    return alpha_(x,t) + (5.0/2.0)*beta_(x,t);
  }
  static P u(P const &x, P const t = 0)
  {
    return (-alpha_(x,t) + (7.0/2.0)*beta_(x,t))/n(x,t);
  }
  static P theta(P const &x, P const t = 0)
  {
    return ((1.0/2.0)*alpha_(x,t) + (25.0/4.0)*beta_(x,t))/n(x,t) - std::pow(u(x,t),2);
  }

  /* build the terms */

  // Term 1
  // -v\cdot\grad_x f for v > 0
  //
  static P e1_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }

  static P e1_g2(P const x, P const time = 0)
  {
    ignore(time);
    return (x > 0.0) ? x : 0.0;
  }

  inline static const partial_term<P> e1_pterm_x =
      partial_term<P>(coefficient_type::div, e1_g1, partial_term<P>::null_gfunc,
                      flux_type::downwind, boundary_condition::periodic,
                      boundary_condition::periodic);

  inline static const partial_term<P> e1_pterm_v = partial_term<P>(
      coefficient_type::mass, e1_g2, partial_term<P>::null_gfunc,
      flux_type::central, boundary_condition::periodic,
      boundary_condition::periodic);

  inline static term<P> const term_e1x = term<P>(false,  // time-dependent
                                                 "E1_x", // name
                                                 {e1_pterm_x},
                                                 imex_flag::imex_explicit);

  inline static term<P> const term_e1v = term<P>(false,  // time-dependent
                                                 "E1_v", // name
                                                 {e1_pterm_v},
                                                 imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_1 = {term_e1x, term_e1v};

  // Term 2
  // -v\cdot\grad_x f for v < 0
  //
  static P e2_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -1.0;
  }

  static P e2_g2(P const x, P const time = 0)
  {
    ignore(time);
    return (x < 0.0) ? x : 0.0;
  }

  inline static const partial_term<P> e2_pterm_x =
      partial_term<P>(coefficient_type::div, e2_g1, partial_term<P>::null_gfunc,
                      flux_type::upwind, boundary_condition::periodic,
                      boundary_condition::periodic);

  inline static const partial_term<P> e2_pterm_v = partial_term<P>(
      coefficient_type::mass, e2_g2, partial_term<P>::null_gfunc,
      flux_type::central, boundary_condition::periodic,
      boundary_condition::periodic);

  inline static term<P> const term_e2x = term<P>(false,  // time-dependent
                                                 "E2_x", // name
                                                 {e2_pterm_x},
                                                 imex_flag::imex_explicit);

  inline static term<P> const term_e2v = term<P>(false,  // time-dependent
                                                 "E2_v", // name
                                                 {e2_pterm_v},
                                                 imex_flag::imex_explicit);

  inline static std::vector<term<P>> const terms_2 = {term_e2x, term_e2v};

  // Term 3
  // v\cdot\grad_v f
  //
  static P i1_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return nu;
  }

  static P i1_g2(P const x, P const time = 0)
  {
    ignore(time);
    return x;
  }

  inline static const partial_term<P> i1_pterm_x = partial_term<P>(
      coefficient_type::mass, i1_g1, partial_term<P>::null_gfunc,
      flux_type::central, boundary_condition::periodic,
      boundary_condition::periodic);

  inline static const partial_term<P> i1_pterm_v =
      partial_term<P>(coefficient_type::div, i1_g2, partial_term<P>::null_gfunc,
                      flux_type::downwind, boundary_condition::dirichlet,
                      boundary_condition::dirichlet);

  inline static term<P> const term_i1x = term<P>(false,  // time-dependent
                                                 "I1_x", // name
                                                 {i1_pterm_x},
                                                 imex_flag::imex_implicit);

  inline static term<P> const term_i1v = term<P>(false,  // time-dependent
                                                 "I1_v", // name
                                                 {i1_pterm_v},
                                                 imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_3 = {term_i1x, term_i1v};

  // Term 4
  // -u\cdot\grad_v f
  //
  static P i2_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return -u(x);
  }

  static P i2_g2(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return nu;
  }

  inline static const partial_term<P> i2_pterm_x = partial_term<P>(
      coefficient_type::mass, i2_g1, partial_term<P>::null_gfunc,
      flux_type::central, boundary_condition::periodic,
      boundary_condition::periodic);

  inline static const partial_term<P> i2_pterm_v =
      partial_term<P>(coefficient_type::div, i2_g2, partial_term<P>::null_gfunc,
                      flux_type::central, boundary_condition::dirichlet,
                      boundary_condition::dirichlet);

  inline static term<P> const term_i2x = term<P>(true,  // time-dependent
                                                 "I2_x", // name
                                                 {i2_pterm_x},
                                                 imex_flag::imex_implicit);

  inline static term<P> const term_i2v = term<P>(false,  // time-dependent
                                                 "I2_v", // name
                                                 {i2_pterm_v},
                                                 imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_4 = {term_i2x, term_i2v};

  // Term 5
  // div_v(th\grad_v f)
  //
  // Split by LDG
  //
  // div_v(th q)
  // q = \grad_v f
  static P i3_g1(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  static P i3_g2(P const x, P const time = 0)
  {
    ignore(time);
    return theta(x) * nu;
  }

  inline static const partial_term<P> i3_pterm_x1 = partial_term<P>(
      coefficient_type::mass, i3_g1, partial_term<P>::null_gfunc,
      flux_type::central, boundary_condition::periodic,
      boundary_condition::periodic);

  inline static const partial_term<P> i3_pterm_x2 = partial_term<P>(
      coefficient_type::mass, i3_g2, partial_term<P>::null_gfunc,
      flux_type::central, boundary_condition::periodic,
      boundary_condition::periodic);

  inline static term<P> const term_i3x = term<P>(true,  // time-dependent
                                                 "I3_x", // name
                                                 {i3_pterm_x1, i3_pterm_x2},
                                                 imex_flag::imex_implicit);

  static P i3_g3(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  static P i3_g4(P const x, P const time = 0)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  inline static const partial_term<P> i3_pterm_v1 =
      partial_term<P>(coefficient_type::div, i3_g3, partial_term<P>::null_gfunc,
                      flux_type::central, boundary_condition::dirichlet,
                      boundary_condition::dirichlet);

  inline static const partial_term<P> i3_pterm_v2 = partial_term<P>(
      coefficient_type::grad, i3_g4, partial_term<P>::null_gfunc,
      flux_type::central, boundary_condition::dirichlet,
      boundary_condition::dirichlet);

  inline static term<P> const term_i3v = term<P>(false,  // time-dependent
                                                 "I3_v", // name
                                                 {i3_pterm_v1, i3_pterm_v2},
                                                 imex_flag::imex_implicit);

  inline static std::vector<term<P>> const terms_5 = {term_i3x, term_i3v};

  inline static term_set<P> const terms_ = {terms_1, terms_2, terms_3, terms_4,
                                            terms_5};

  //////////////////////////////////////////////////////////////////
  // Solution
  ////////////////////////////////////////////////////////////////// 
                                     
  inline static md_func_type<P> soln_0 = {soln_dim_x_0,soln_dim_v_0};
  inline static md_func_type<P> soln_1 = {soln_dim_x_1,soln_dim_v_1};

  static P exact_time(P const time) { ignore(time); return 1.0; }

  inline static std::vector<md_func_type<P>> const exact_vector_funcs_ = {soln_0,soln_1};
  inline static scalar_func<P> const exact_scalar_func_                = {exact_time};

  //////////////////////////////////////////////////////////////////
  // Sources
  //////////////////////////////////////////////////////////////////

  static P sc_time(P const time) { ignore(time); return 1.0; }

  // d_tf
  static fk::vector<P> source_dim_x_0(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = dt_alpha_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_0(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = Mxwl_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_x_1(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = dt_beta_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_1(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = onepv2_(x[i],t)*Mxwl_(x[i],t);
    }
    return fx;
  }

  // vd_xf
  static fk::vector<P> source_dim_x_2(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = dx_alpha_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_2(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = x[i]*Mxwl_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_x_3(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = dx_beta_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_3(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = x[i]*onepv2_(x[i],t)*Mxwl_(x[i],t);
    }
    return fx;
  }

  // -nu*f
  static fk::vector<P> source_dim_x_4(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = -nu*alpha_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_4(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = Mxwl_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_x_5(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = -nu*beta_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_5(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = onepv2_(x[i],t)*Mxwl_(x[i],t);
    }
    return fx;
  }

  // -nu*v*d_vf
  static fk::vector<P> source_dim_x_6(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = -nu*alpha_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_6(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = -2*x[i]*(x[i]-1.0)*Mxwl_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_x_7(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = -nu*beta_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_7(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = (2.0*x[i]-2.0*(1.0+x[i]*x[i])*(x[i]-1.0))*x[i]*Mxwl_(x[i],t);
    }
    return fx;
  }

  // nu*u*d_vf
  static fk::vector<P> source_dim_x_8(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = nu*alpha_(x[i],t)*u(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_8(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = -2.0*(x[i]-1.0)*Mxwl_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_x_9(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = nu*beta_(x[i],t)*u(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_9(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = (2.0*x[i]-2.0*(1.0+x[i]*x[i])*(x[i]-1.0))*Mxwl_(x[i],t);
    }
    return fx;
  }

  // -nu*theta*d_vvf
  static fk::vector<P> source_dim_x_10(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = -nu*alpha_(x[i],t)*theta(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_10(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = (-2.0+4.0*std::pow(x[i]-1.0,2))*Mxwl_(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_x_11(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = -nu*beta_(x[i],t)*theta(x[i],t);
    }
    return fx;
  }
  static fk::vector<P> source_dim_v_11(fk::vector<P> const x, P const t = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i=0; i < x.size(); i++)
    {
      fx[i] = (2.0-8.0*x[i]*(x[i]-1.0)-2*onepv2_(x[i])+4*onepv2_(x[i])*std::pow(x[i]-1.0,2))*Mxwl_(x[i],t);
    }
    return fx;
  }

inline static source<P> const source0_  = source<P>({source_dim_x_0 , source_dim_v_0 }, sc_time);
inline static source<P> const source1_  = source<P>({source_dim_x_1 , source_dim_v_1 }, sc_time);
inline static source<P> const source2_  = source<P>({source_dim_x_2 , source_dim_v_2 }, sc_time);
inline static source<P> const source3_  = source<P>({source_dim_x_3 , source_dim_v_3 }, sc_time);
inline static source<P> const source4_  = source<P>({source_dim_x_4 , source_dim_v_4 }, sc_time);
inline static source<P> const source5_  = source<P>({source_dim_x_5 , source_dim_v_5 }, sc_time);
inline static source<P> const source6_  = source<P>({source_dim_x_6 , source_dim_v_6 }, sc_time);
inline static source<P> const source7_  = source<P>({source_dim_x_7 , source_dim_v_7 }, sc_time);
inline static source<P> const source8_  = source<P>({source_dim_x_8 , source_dim_v_8 }, sc_time);
inline static source<P> const source9_  = source<P>({source_dim_x_9 , source_dim_v_9 }, sc_time);
inline static source<P> const source10_ = source<P>({source_dim_x_10, source_dim_v_10}, sc_time);
inline static source<P> const source11_ = source<P>({source_dim_x_11, source_dim_v_11}, sc_time);

inline static std::vector<source<P>> const sources_ = { source0_, source1_, source2_, source3_,
                                                        source4_, source5_, source6_, source7_,
                                                        source8_, source9_,source10_,source11_ };

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

};

} // namespace asgard
