#include "equilibrium_solve.hpp"
#include "odepp/integrators/forward_euler.hpp"
#include "odepp/ode_solve.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
constexpr long double operator"" _yrs(long double years) {
  return years * 365.0;
}

constexpr long double operator"" _days(long double x) { return x; };

int main() {
  static constexpr auto K_max = 1;

  auto y0 = std::array<double, 6 * (K_max + 1)>{2000.0};
  const auto params = equilibrium::Parameters{.eir = 1.0 / (1.0_yrs),
                                              .zeta = 1.0,
                                              .f = 1.0 / (41.0_days),
                                              .a0 = 8.0_yrs,
                                              .r_PCR = 1.0 / (52.0_days),
                                              .r_LM = 1.0 / (16.0_days),
                                              .r_D = 1.0 / (5.0_days),
                                              .r_T = 1.0 / (1.0_days),
                                              .r_P = 1.0 / (6.0_days),
                                              .gamma_L = 1.0 / (383.0_days),
                                              .mu_h = 1.0 / (22.5_yrs),
                                              .phiD = 0.0,
                                              .phiLM = 0.0,
                                              .chi = 0.0};
  auto f = [params](auto age, const auto &y) {
    return equilibrium::residual_equation<std::array>(age, y, params);
  };
  auto [age_out, yout] = odepp::ode_solve(odepp::integrator::forward_euler, f,
                                          0.0, 80.0_yrs, 1.0_days, y0);

  std::filesystem::create_directories("output");
  auto file = std::ofstream("output/data.csv");
  if (!file.is_open()) {
    throw;
  }

  for (auto x : f(0.0, y0)) {
    std::cout << x << std::endl;
  }

  file << "age,state,hypnozoites,number\n";
  for (const auto &[age, state] : std::views::zip(age_out, yout)) {
    for (auto k = 0; k <= K_max; ++k) {
      file << age << ',';
      file << "S," << k << ',' << state.at(6 * k) << '\n';
      file << age << ',';
      file << "IPCR," << k << ',' << state.at(6 * k + 1) << '\n';
      file << age << ',';
      file << "ILM," << k << ',' << state.at(6 * k + 2) << '\n';
      file << age << ',';
      file << "ID," << k << ',' << state.at(6 * k + 3) << '\n';
      file << age << ',';
      file << "T," << k << ',' << state.at(6 * k + 4) << '\n';
      file << age << ',';
      file << "P," << k << ',' << state.at(6 * k + 5) << '\n';
    }
  }
  file.close();
}
