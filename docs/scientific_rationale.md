# Scientific Rationale & Physiological Foundations

## 1. Physical Principles of Remote Photoplethysmography (rPPG)

Remote Photoplethysmography (rPPG) is an optical technique that measures micro-vascular blood volume pulsations beneath facial dermis using ambient illumination and standard RGB image sensors.

### 1.1 The Beer-Lambert Law & Hemoglobin Absorption
When ambient light strikes skin tissue, a fraction undergoes specular reflection at the stratum corneum, while the remainder penetrates into subcutaneous capillary beds. Oxygenated hemoglobin ($\text{HbO}_2$) and deoxygenated hemoglobin ($\text{Hb}$) exhibit peak optical absorption in the green spectrum ($\lambda \approx 520\text{--}570\text{ nm}$):

$$\Delta I(t) = I_0 e^{-\epsilon C d(t)}$$

where:
- $\epsilon$ is the molar extinction coefficient of hemoglobin,
- $C$ is the concentration of absorbing chromophores,
- $d(t) = d_0 + \Delta d(t)$ is the dynamic optical path length modulated by systolic pulse waves.

Because the green channel has the highest ratio of hemoglobin absorptance to melanin scattering, spatial averaging of green pixel values yields the highest pulsatile signal-to-noise ratio.

---

## 2. Algorithmic Formulations: POS vs. CHROM

Ambient lighting variations, subject movement, and head tilts introduce illumination noise orders of magnitude larger than capillary blood pulsation ($\approx 0.1\%\text{--}1.5\%$ of ambient intensity).

### 2.1 The Plane-Orthogonal-to-Skin (POS) Formulation
Developed by Wang et al. (IEEE TBME 2017), POS defines an optical skin reflection model:
$$\mathbf{C}(t) = I(t) \cdot \left( \mathbf{u}_s \cdot s(t) + \mathbf{u}_p \cdot p(t) \right) + \mathbf{v}_{\text{motion}}(t)$$

where $\mathbf{u}_s$ is the stationary skin-tone unit vector, and $\mathbf{u}_p$ is the pulsatile physiological absorption vector. By projecting temporal signals onto a plane orthogonal to $\mathbf{u}_s$, intensity variations caused by motion and shadow are minimized.

---

## 3. Blood Pressure Estimation via Pulse Wave Analysis

Arterial Blood Pressure (BP) correlates with vascular elasticity and cardiac output. In non-contact systems:
1. **Pulse Waveform Morphology:** Systolic upstroke slope, augmentation index ($A_r$), and dicrotic reflection time reflect arterial compliance (Bramwell-Hill equation).
2. **Moens-Korteweg Equation:**
   $$\text{PWV} = \sqrt{\frac{E \cdot h}{2 r \rho}}$$
   where $E$ is Young's modulus of arterial wall elasticity (which increases with blood pressure), $h$ is wall thickness, and $\rho$ is blood density.
3. **Limitation:** While relative changes in pulse wave transit time provide statistical correlation with SBP/DBP, absolute estimation requires individual calibration against cuff sphygmomanometers.

---

## 4. Critical Scientific Reality: Glucose and Cholesterol

> [!WARNING]
> Non-contact visible-light cameras cannot directly measure systemic molecular concentrations of blood glucose or total cholesterol through skin reflectance.

- **Spectral Absence:** Glucose and cholesterol lack distinct absorption bands in the visible spectrum ($400\text{--}700\text{ nm}$). Non-invasive glucose research requires designated mid-infrared or near-infrared spectroscopy ($1000\text{--}2500\text{ nm}$), Raman spectroscopy, or radio-frequency dielectric sensing.
- **Indirect Autonomic Correlates:** Some research explores whether postprandial glucose shifts subtly affect autonomic vascular tone or heart rate variability (HRV). However, these correlations are non-specific and influenced by stress, hydration, temperature, and circadian rhythms.
- **Safety Mandate:** FaceVital AI explicitly flags glucose and cholesterol as **experimental research models** with placeholder status unless calibrated in controlled clinical laboratory protocols.

---

## 5. References

1. **Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).** Algorithmic Principles of Remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
2. **de Haan, G., & Jeanne, V. (2013).** Robust Pulse Rate From Chrominance-Based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886.
3. **Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).** Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express*, 18(10), 10762–10774.
4. **Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008).** Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.
