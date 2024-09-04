#ifndef MWHITE_EQUILIBRIUM_HPP
#define MWHITE_EQUILIBRIUM_HPP
#include <span>
namespace equilibrium {
template <std::size_t offset, std::size_t stride, typename T>
struct index_adaptor {
  T &data;
  inline auto operator[](const std::size_t &index) -> typename T::value_type & {
    return data[offset + stride * index];
  };
};

struct Parameters {
  double eir;
  double zeta;
  double f;
  double a0;
  double r_PCR;
  double r_LM;
  double r_D;
  double r_T;
  double r_P;
  double gamma_L;
  double mu_h;
  double phiD;
  double phiLM;
  double chi;
};

template <template <typename, std::size_t> typename Container,
          std::floating_point RealType, std::size_t N>
auto residual_equation([[maybe_unused]] std::floating_point auto t,
                       std::array<RealType, N> y, const Parameters &params) {
  const auto [eir, zeta, f, a0, r_PCR, r_LM, r_D, r_T, r_P, gamma_L, mu_h, phiD,
              phiLM, chi] = params;

  static constexpr auto K_max = N / 6 - 1;
  auto output = Container<RealType, N>();
  auto Sout = index_adaptor<0, 6, decltype(output)>{output};
  auto IPCRout = index_adaptor<1, 6, decltype(output)>{output};
  auto ILMout = index_adaptor<2, 6, decltype(output)>{output};
  auto IDout = index_adaptor<3, 6, decltype(output)>{output};
  auto Tout = index_adaptor<4, 6, decltype(output)>{output};
  auto Pout = index_adaptor<5, 6, decltype(output)>{output};
  auto S = index_adaptor<0, 6, decltype(y)>{y};
  auto IPCR = index_adaptor<1, 6, decltype(y)>{y};
  auto ILM = index_adaptor<2, 6, decltype(y)>{y};
  auto ID = index_adaptor<3, 6, decltype(y)>{y};
  auto T = index_adaptor<4, 6, decltype(y)>{y};
  auto P = index_adaptor<5, 6, decltype(y)>{y};

  auto lambda = eir * zeta;
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
