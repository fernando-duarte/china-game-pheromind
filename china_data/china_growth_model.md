# Open-Economy Growth Model for China (1980–2025)

## Variables

### Endogenous

| Symbol       | Definition                      | Units       |
| :----------- | :------------------------------ | :---------- |
| $Y_t$        | Real GDP                        | bn USD      |
| $K_t$        | Physical capital stock          | bn USD      |
| $A_t$        | Total factor productivity (TFP) | index       |
| $X_t$        | Exports                         | bn USD      |
| $M_t$        | Imports                         | bn USD      |
| $NX_t$       | Net exports                     | bn USD      |
| $C_t$        | Consumption                     | bn USD      |
| $I_t$        | Investment                      | bn USD      |
| $openness_t$ | (Exports + Imports) / GDP       | fraction    |


### Exogenous

| Symbol         | Definition                                    | Units                                 |
| :------------- | :-------------------------------------------- | :------------------------------------ |
| $e_t$        | Nominal exchange rate           | CNY per USD |
| $L_t$        | Labor force                     | million     |
| $fdi\_ratio_t$ | FDI inflows \/ GDP                            | fraction                              |
| $Y^*_t$        | Foreign income                                | index (1980 = 1000)                   |
| $H_t$          | Human capital index                           | index (2015 = Penn World Table value) |
| $G_t$          | Government spending                           | bn USD                                |
| $T_t$          | Taxes                           | bn USD                                |
| $s_t$          | Saving rate                           | fraction                                |

### Parameters

| Symbol                          | Definition                                   | Units    | Value       |
| :------------------------------ | :------------------------------------------- | :------- | :---------- |
| $\alpha$                        | Capital share in production                  | unitless | $0.30$      |
| $\delta$                        | Depreciation rate                            | per year | $0.10$      |
| $g$                             | Baseline TFP growth rate                     | per year | $0.005$     |
| $\theta$                        | Openness contribution to TFP growth          | unitless | $0.1453$    |
| $\phi$                          | FDI contribution to TFP growth               | unitless | $0.10$      |
| $K_0$                           | Initial level of physical capital (1980)     | bn USD   | $2050.10$   |
| $X_0$                           | Initial level of exports (1980)              | bn USD   | $18.10$     |
| $M_0$                           | Initial level of imports (1980)              | bn USD   | $14.50$     |
| $\varepsilon_x,\ \varepsilon_m$ | Exchange‐rate elasticities (exports/imports) | unitless | $1.5,\ 1.2$ |
| $\mu_x,\ \mu_m$                 | Income elasticities (exports/imports)        | unitless | $1.0,\ 1.0$ |


## Paths of exogenous variables

| Year | $\tilde e_t$ | $fdi\_ratio_t$ | $Y^*_t$ | $H_t$ |
| ---: | -----------: | -------------: | ------: | ----: |
| 1980 |         0.78 |          0.001 | 1000.00 |  1.58 |
| 1985 |         1.53 |          0.001 | 1159.27 |  1.77 |
| 1990 |         2.48 |           0.02 | 1343.92 |  1.80 |
| 1995 |         4.34 |           0.02 | 1557.97 |  2.02 |
| 2000 |         5.23 |           0.02 | 1806.11 |  2.24 |
| 2005 |         4.75 |           0.02 | 2093.78 |  2.43 |
| 2010 |         5.61 |           0.02 | 2427.26 |  2.61 |
| 2015 |         7.27 |           0.02 | 2813.86 |  2.60 |
| 2020 |         7.00 |           0.02 | 3262.04 |  6.71 |
| 2025 |         6.41 |           0.02 | 3781.60 |  6.49 |

## Model Equations

- **Production:**
  $$Y_t = A_t\,K_t^{\alpha}\,(L_t\,H_t)^{1-\alpha}$$

- **Capital accumulation:**
  $$K_{t+1} = (1-\delta)\,K_t + I_t$$
  $$K_0 \text{ given}$$

- **Labor force:**
  $$L_{t+1} = (1+n) L_t$$

- **TFP:**
The law of motion for technology \(A_t\) with spillover effects from openness and FDI can be written generally as:

    $$
      A_{t+1} = A_t \left(1 + g + f(\text{spillovers}_t)\right),
    $$

    where $f(\text{spillovers}_t)$ captures the effect of trade openness, foreign direct investment, and other external factors influencing technology growth, which we model as in in Barro and Sala-i-Martin (see Economic Growth, MIT Press, 2nd edition, Chapter 8, 2004, isbn: 9780262025539):

    $$
      A_{t+1} = A_t \left(1 + g + \theta\, \text{openness}_t + \phi\, \text{fdi\_ratio}_t \right),
    $$

- **Exports:**
  $$
    X_t = X_0\Bigl(\tfrac{e_t}{e_{1980}}\Bigr)^{\varepsilon_x}
      \Bigl(\tfrac{Y^*_t}{Y^*_{1980}}\Bigr)^{\mu_x}
  $$

- **Imports:**
  $$
    M_t = M_0\Bigl(\tfrac{e_t}{e_{1980}}\Bigr)^{-\varepsilon_m}
      \Bigl(\tfrac{Y_t}{Y_{1980}}\Bigr)^{\mu_m}
  $$

- **Net exports:**
  $$
    NX_t = X_t - M_t
  $$

- **Saving:**
  $$S_t = Y_t - C_t - G_t = I_t + NX_t$$

- **Private Saving:**
  $$S^{\mathrm{priv}}_t = Y_t - T_t - C_t$$

- **Public Saving:**
  $$S^{\mathrm{pub}}_t = T_t - G_t$$

- **Saving Rate:**
  $$s_t = \frac{S_t}{Y_t} = \frac{I_t + NX_t}{Y_t} = 1 - \frac{C_t + G_t}{Y_t}$$

- **Consumption:**
  $$C_t = (1 - s_t)Y_t - G_t$$

- **Investment:**
  $$I_t = s_t Y_t - NX_t$$

- **Openness ratio:**
  $$openness_t = \frac{X_t + M_t}{Y_t}$$

## Computation Steps for Each Round

### Read values

1. Read values of parameters and paths of exogenous variables. These are known before any computation starts.

### Compute variables for t=1,2,...

14. Compute TFP:
    $$
      A_{t} = A_{t-1}
        (1 + g
          + \theta\,openness_t
          + \phi\,fdi\_ratio_t
        )
    $$

2. Compute output/production:
   $$ Y_t = A_t K_t^{\alpha} (L_t\,H_t)^{1-\alpha} $$
5. Compute nominal exchange rate:
   $$ e_t = x_t \tilde e_t $$
6. Compute exports:
   $$
     X_t = X_0\Bigl(\tfrac{e_t}{e_{0}}\Bigr)^{\varepsilon_x}
       \Bigl(\tfrac{Y^*_t}{Y^*_{0}}\Bigr)^{\mu_x}.
   $$
7. Compute imports:
   $$
     M_t = M_0\Bigl(\tfrac{e_t}{e_{0}}\Bigr)^{-\varepsilon_m}
       \Bigl(\tfrac{Y_t}{Y_{0}}\Bigr)^{\mu_m}.
   $$
8. Compute net exports:
   $$ NX_t = X_t - M_t $$
9. Compute openness ratio:
   $$
     openness_t = \frac{X_t + M_t}{Y_t}
   $$
10. Compute consumption:
    $$ C_t = (1-s) Y_t $$

11. Compute investment:
    $$ I_t = s Y_t - NX_t $$

### Compute next period's variable values

13. Compute next period's capital:
    $$ K\_{t+1} = (1-\delta) K_t + I_t $$


