#ifndef MWHITE_EQUILIBRIUM_HPP
#define MWHITE_EQUILIBRIUM_HPP
#include <cmath>
#include <concepts>
#include <span>
#include <type_traits>
namespace equilibrium {
template <std::size_t offset, std::size_t stride, typename T>
struct index_adaptor {
  T &data;
  inline auto operator[](const std::size_t &index)
      -> std::conditional_t<
          std::is_const_v<T>,
          typename std::remove_reference_t<T>::const_reference,
          typename std::remove_reference_t<T>::reference> {
    return data[offset + stride * index];
  };
};

struct HillFuncParameters {
  double c50;
  double min;
  double max;
  double kappa;
};

struct ImmunityParameters {
  double Pmat;
  double u_par;
  double u_clin;
  double mu_M;
  double mu_P;
  double mu_C;
};

struct TransmissionParameters {
  double eir;
  double zeta;
  double f;
  double rho;
  double a0;
  double r_LM;
  double r_D;
  double r_T;
  double r_P;
  double gamma_L;
  double mu_h;
  double chi;
};

struct Parameters {
  TransmissionParameters transmission_params;
  ImmunityParameters immune_params;
  HillFuncParameters phiLM;
  HillFuncParameters phiD;
  HillFuncParameters dPCR;
};

template <std::floating_point RealType>
auto hill_func(std::floating_point auto x, std::floating_point auto y,
               const HillFuncParameters &p) {
  auto denom = [](auto x, auto y, auto c50, auto kappa) {
    return 1.0 + std::pow((x + y) / c50, kappa);
  };
  return p.min + (p.max - p.min) / denom(x, y, p.c50, p.kappa);
};

template <std::floating_point RealType>
auto immunity_rhs_equation([[maybe_unused]] std::floating_point auto age,
                           const std::floating_point auto &y,
                           const std::floating_point auto exposure,
                           const std::floating_point auto decay_rate) {
  return exposure - decay_rate * y;
};

template <std::floating_point RealType>
auto all_immunity_rhs_equation(std::floating_point auto age,
                               std::span<const RealType> y,
                               std::span<RealType> y_out,
                               std::floating_point auto lambda,
                               const ImmunityParameters &immune_params) {
  const auto &[Pmat, u_par, u_clin, mu_M, mu_P, mu_C] = immune_params;
  // Antibody model for immunity.
  auto exposure_parasite = lambda / (lambda * u_par + 1.0);
  auto exposure_clinical = lambda / (lambda * u_clin + 1.0);
  y_out[0] = immunity_rhs_equation<RealType>(age, y[0], 0.0, mu_M);
  y_out[1] =
      immunity_rhs_equation<RealType>(age, y[1], exposure_parasite, mu_P);
  y_out[2] = immunity_rhs_equation<RealType>(age, y[2], 0.0, mu_M);
  y_out[3] =
      immunity_rhs_equation<RealType>(age, y[3], exposure_clinical, mu_C);
};

template <template <typename, std::size_t> typename Container,
          std::floating_point RealType, std::size_t N>
auto residual_equation(std::floating_point auto age,
                       const Container<RealType, N> &y,
                       const Parameters &params) {

  static constexpr auto K_max = N / 6 - 1;
  static_assert(K_max > 0);

  auto output = Container<RealType, N>();
  auto Sout = index_adaptor<0 + 4, 6, decltype(output)>{output};
  auto IPCRout = index_adaptor<1 + 4, 6, decltype(output)>{output};
  auto ILMout = index_adaptor<2 + 4, 6, decltype(output)>{output};
  auto IDout = index_adaptor<3 + 4, 6, decltype(output)>{output};
  auto Tout = index_adaptor<4 + 4, 6, decltype(output)>{output};
  auto Pout = index_adaptor<5 + 4, 6, decltype(output)>{output};

  const auto &AP_mat = y[0];
  const auto &AP = y[1];
  const auto &AC_mat = y[2];
  const auto &AC = y[3];

  auto S = index_adaptor<0 + 4, 6, const Container<RealType, N>>{y};
  auto IPCR = index_adaptor<1 + 4, 6, const Container<RealType, N>>{y};
  auto ILM = index_adaptor<2 + 4, 6, const Container<RealType, N>>{y};
  auto ID = index_adaptor<3 + 4, 6, const Container<RealType, N>>{y};
  auto T = index_adaptor<4 + 4, 6, const Container<RealType, N>>{y};
  auto P = index_adaptor<5 + 4, 6, const Container<RealType, N>>{y};

  const auto &[eir, zeta, f, rho, a0, r_LM, r_D, r_T, r_P, gamma_L, mu_h, chi] =
      params.transmission_params;

  auto psi = (1.0 - rho * std::exp(-age / a0)) /
             (1.0 - a0 * mu_h * rho / (1 + a0 * mu_h));
  auto lambda = psi * zeta * eir;

  auto immunity_in = std::span<const RealType>(y.begin(), 4);
  auto immunity_out = std::span<RealType>(output.begin(), 4);

  all_immunity_rhs_equation<RealType>(age, immunity_in, immunity_out, lambda,
                                      params.immune_params);

  const auto phiLM = hill_func<RealType>(AP, AP_mat, params.phiLM);
  const auto phiD = hill_func<RealType>(AC, AC_mat, params.phiD);
  const auto r_PCR = 1.0 / hill_func<RealType>(AP, AP_mat, params.dPCR);

  // First row.
  auto k = 0U;
  Sout[k] = -lambda * S[k] + r_PCR * IPCR[k] + r_P * P[k] +
            gamma_L * (k + 1) * S[k + 1] - mu_h * S[k];
  IPCRout[k] = -lambda * IPCR[k] - r_PCR * IPCR[k] + r_LM * ILM[k] +
               +gamma_L * (k + 1) * IPCR[k + 1] - mu_h * IPCR[k];
  ILMout[k] = -lambda * ILM[k] - r_LM * ILM[k] + r_D * ID[k] +
              gamma_L * (k + 1) * ILM[k + 1] - mu_h * ILM[k];
  IDout[k] = -lambda * ID[k] - r_D * ID[k] + gamma_L * (k + 1) * ID[k + 1] -
             mu_h * ID[k];
  Tout[k] =
      -lambda * T[k] - r_T * T[k] + gamma_L * (k + 1) * T[k + 1] - mu_h * T[k];
  Pout[k] = -lambda * P[k] + r_T * T[k] - r_P * P[k] +
            gamma_L * (k + 1) * P[k + 1] - mu_h * P[k];
  for (k = 1U; k < K_max; ++k) {
    // This is not optimised in any way shape or form. But it does look like
    // what is in MWhites paper so that might make it more readable.
    Sout[k] = -(lambda + f * k) * S[k] + r_PCR * IPCR[k] + r_P * P[k] -
              gamma_L * k * S[k] + gamma_L * (k + 1) * S[k + 1] - mu_h * S[k];
    IPCRout[k] = -(lambda + f * k) * IPCR[k] - r_PCR * IPCR[k] + r_LM * ILM[k] +
                 lambda * (1 - phiLM) * (S[k - 1] + IPCR[k - 1]) +
                 f * k * (1 - phiLM) * (S[k] + IPCR[k]) -
                 gamma_L * k * IPCR[k] + gamma_L * (k + 1) * IPCR[k + 1] -
                 mu_h * IPCR[k];
    ILMout[k] = -(lambda + f * k) * ILM[k] - r_LM * ILM[k] + r_D * ID[k] +
                lambda * (1 - phiD) *
                    (phiLM * S[k - 1] + phiLM * IPCR[k - 1] + ILM[k - 1]) +
                f * k * (1 - phiD) * (phiLM * (S[k] + IPCR[k]) + ILM[k]) -
                gamma_L * k * ILM[k] + gamma_L * (k + 1) * ILM[k + 1] -
                mu_h * ILM[k];
    IDout[k] = -lambda * (ID[k] - ID[k - 1]) - r_D * ID[k] +
               lambda * phiD * (1 - chi) *
                   (phiLM * (S[k - 1] + IPCR[k - 1]) + ILM[k - 1]) +
               f * k * phiD * (1 - chi) * (phiLM * (S[k] + IPCR[k]) + ILM[k]) -
               gamma_L * k * ID[k] + gamma_L * (k + 1) * ID[k + 1] -
               mu_h * ID[k];
    Tout[k] =
        -lambda * (T[k] - T[k - 1]) - r_T * T[k] +
        lambda * phiD * chi * (phiLM * (S[k - 1] + IPCR[k - 1]) + ILM[k - 1]) +
        f * k * phiD * chi * (phiLM * (S[k] + IPCR[k]) + ILM[k]) -
        gamma_L * k * T[k] + gamma_L * (k + 1) * T[k + 1] - mu_h * T[k];
    Pout[k] = -lambda * (P[k] - P[k - 1]) + r_T * T[k] - r_P * P[k] -
              gamma_L * k * P[k] + gamma_L * (k + 1) * P[k + 1] - mu_h * P[k];
  }

  k = K_max;
  Sout[k] = -(lambda + f * k) * S[k] + r_PCR * IPCR[k] + r_P * P[k] -
            gamma_L * k * S[k] + -mu_h * S[k];
  IPCRout[k] =
      -(lambda + f * k) * IPCR[k] - r_PCR * IPCR[k] + r_LM * ILM[k] +
      lambda * (1 - phiLM) * (S[k - 1] + S[k] + IPCR[k - 1] + IPCR[k]) +
      f * k * (1 - phiLM) * (S[k] + IPCR[k]) - gamma_L * k * IPCR[k] +
      -mu_h * IPCR[k];
  ILMout[k] = -(lambda + f * k) * ILM[k] - r_LM * ILM[k] + r_D * ID[k] +
              lambda * (1 - phiD) *
                  (phiLM * (S[k - 1] + S[k] + IPCR[k - 1] + IPCR[k]) +
                   ILM[k - 1] + ILM[k]) +
              f * k * (1 - phiD) * (phiLM * (S[k] + IPCR[k]) + ILM[k]) -
              gamma_L * k * ILM[k] - mu_h * ILM[k];

  IDout[k] = -lambda * (ID[k] - ID[k - 1]) - r_D * ID[k] +
             lambda * phiD * (1 - chi) *
                 (phiLM * (S[k - 1] + S[k] + IPCR[k - 1] + IPCR[k]) +
                  ILM[k - 1] + ILM[k]) +
             f * k * phiD * (1 - chi) * (phiLM * (S[k] + IPCR[k]) + ILM[k]) -
             gamma_L * k * ID[k] - mu_h * ID[k];
  Tout[k] = -lambda * (-T[k - 1]) - r_T * T[k] +
            lambda * phiD * chi *
                (phiLM * (S[k - 1] + S[k] + IPCR[k - 1] + IPCR[k]) +
                 ILM[k - 1] + ILM[k]) +
            f * k * phiD * chi * (phiLM * (S[k] + IPCR[k]) + ILM[k]) -
            gamma_L * k * T[k] - mu_h * T[k];
  Pout[k] = -lambda * (-P[k - 1]) + r_T * T[k] - r_P * P[k] -
            gamma_L * k * P[k] - mu_h * P[k];
  return output;
}
} // namespace equilibrium
#endif // !MWHITE_EQUILIBRIUM_HPP
