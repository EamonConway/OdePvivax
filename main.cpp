#include "equilibrium_solve.hpp"
#include "odepp/integrators/forward_euler.hpp"
#include "odepp/ode_solve.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>

// Helpers to make sure that we are using the correct units.
constexpr long double operator"" _years(long double years) {
  return years * 365.0;
}
constexpr long double operator"" _days(long double x) { return x; };
int main() {
  static constexpr auto K_max = 20;
  static constexpr auto Npop = 10000;
  static constexpr auto mothers_age = 20.0_years;
  const auto transmission_params =
      equilibrium::TransmissionParameters{.eir = 100.0 / 365.0,
                                          .zeta = 1.0,
                                          .f = 1.0 / (41.0_days),
                                          .rho = 0.85,
                                          .a0 = 8.0_years,
                                          .r_LM = 1.0 / (16.0_days),
                                          .r_D = 1.0 / (5.0_days),
                                          .r_T = 1.0 / (1.0_days),
                                          .r_P = 1.0 / (6.0_days),
                                          .gamma_L = 1.0 / (383.0_days),
                                          .mu_h = 1.0 / (22.5_years),
                                          .chi = 0.3};

  const auto params = equilibrium::Parameters{
      .transmission_params = transmission_params,
      .immune_params = {.Pmat = 0.31,
                        .u_par = 42.4_days,
                        .u_clin = 4.33_days,
                        .mu_M = 1.0 / (49.9_days),
                        .mu_P = 1.0 / (10.0_years),
                        .mu_C = 1.0 / (30.0_years)},
      .phiLM = {.c50 = 18.8, .min = 0.011, .max = 0.93, .kappa = 3.37},
      .phiD = {.c50 = 24.5, .min = 0.006, .max = 0.96, .kappa = 5.63},
      .dPCR = {.c50 = 9.9, .min = 10.0, .max = 52.6, .kappa = 3.82}};

  auto initial_immunity = [](const auto &params) {
    auto immunity_model = [](auto age, const auto &y, const auto &params) {
      const auto &[eir, zeta, f, rho, a0, r_LM, r_D, r_T, r_P, gamma_L, mu_h,
                   chi] = params.transmission_params;
      auto psi = (1.0 - rho * std::exp(-age / a0)) /
                 (1.0 - a0 * mu_h * rho / (1 + a0 * mu_h));
      auto lambda = psi * zeta * eir;

      auto immunity = std::array{0.0, 0.0, 0.0, 0.0};
      all_immunity_rhs_equation<double>(age, y, immunity, lambda,
                                        params.immune_params);
      return immunity;
    };

    // Solve the immunity equations first to get the immunity of mothers.
    auto [immune_age, immunity_value] = odepp::ode_solve(
        odepp::integrator::forward_euler, immunity_model, 0.0, mothers_age,
        1.0_days, std::array{0.0, 0.0, 0.0, 0.0}, params);

    return immunity_value.back();
  };

  const auto immunity_value = initial_immunity(params);

  // Define the residual equation (its a templated function so has to be
  // called somewhere to be instantiated)
  auto f = [params](auto age, const auto &y) {
    return equilibrium::residual_equation(age, y, params);
  };

  // Update the initial condition based upon mother immunity levels.
  auto y0 = std::array<double, 6 * (K_max + 1) + 4>{0.0};
  y0[0] = params.immune_params.Pmat * immunity_value[1];
  y0[2] = params.immune_params.Pmat * immunity_value[3];
  y0[4] = params.transmission_params.mu_h * Npop;

  // Solve the ode
  auto [age_out, yout] = odepp::ode_solve(odepp::integrator::forward_euler, f,
                                          0.0, 80.0_years, 1.0_days, y0);

  std::filesystem::create_directories("output");
  auto file = std::ofstream("output/data.csv");
  if (!file.is_open()) {
    throw;
  }

  file << "age,state,hypnozoites,number\n";
  for (const auto &[age, state] : std::views::zip(age_out, yout)) {
    file << age << ',' << "ap_mat,0," << state.at(0) << '\n';
    file << age << ',' << "ap,0," << state.at(1) << '\n';
    file << age << ',' << "ac_mat,0," << state.at(2) << '\n';
    file << age << ',' << "ac,0," << state.at(3) << '\n';
    for (auto k = 0; k <= K_max; ++k) {
      file << age << ',';
      file << "S," << k << ',' << state.at(6 * k + 4) << '\n';
      file << age << ',';
      file << "IPCR," << k << ',' << state.at(6 * k + 1 + 4) << '\n';
      file << age << ',';
      file << "ILM," << k << ',' << state.at(6 * k + 2 + 4) << '\n';
      file << age << ',';
      file << "ID," << k << ',' << state.at(6 * k + 3 + 4) << '\n';
      file << age << ',';
      file << "T," << k << ',' << state.at(6 * k + 4 + 4) << '\n';
      file << age << ',';
      file << "P," << k << ',' << state.at(6 * k + 5 + 4) << '\n';
    }
  }
  file.close();
}
