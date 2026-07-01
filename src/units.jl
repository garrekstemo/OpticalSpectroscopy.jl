"""
Unit parsing and conversions for spectroscopy data.

Uses Unitful.jl and PhysicalConstants.jl for dimensioned quantities.
"""

# ============================================================================
# Physical constants for spectroscopy conversions (CODATA2022)
# ============================================================================

const hc_eVnm = Unitful.ustrip(Unitful.uconvert(u"eV*nm", h * c_0))
const hc_eVcm = Unitful.ustrip(Unitful.uconvert(u"eV*cm", h * c_0))
const ħ_eVfs = Unitful.ustrip(Unitful.uconvert(u"eV*fs", ħ))
const ħ_meVps = Unitful.ustrip(Unitful.uconvert(u"meV*ps", ħ))

# ============================================================================
# Shared numerical constants
# ============================================================================

# Gaussian FWHM = FWHM_FACTOR · σ
const FWHM_FACTOR = 2 * sqrt(2 * log(2))  # ≈ 2.355

# Robust scale from the median absolute deviation: σ ≈ MAD_TO_SIGMA · MAD
# (1/Φ⁻¹(0.75), the normal-consistency factor)
const MAD_TO_SIGMA = 1.4826022185056018

# ============================================================================
# Wavenumber <-> Wavelength conversions
# ============================================================================

"""
    wavenumber_to_wavelength(ν̃; output_unit=u"nm")

Convert wavenumber (cm⁻¹) to wavelength.
"""
function wavenumber_to_wavelength(ν̃::Real; output_unit=Unitful.nm)
    ν̃ <= 0 && throw(ArgumentError("Wavenumber must be positive"))
    λ_cm = 1 / ν̃
    λ_nm = λ_cm * 1e7
    return Unitful.uconvert(output_unit, λ_nm * Unitful.nm)
end

function wavenumber_to_wavelength(ν̃::Unitful.Quantity; output_unit=Unitful.nm)
    ν̃_cm = Unitful.ustrip(Unitful.uconvert(Unitful.cm^-1, ν̃))
    return wavenumber_to_wavelength(ν̃_cm; output_unit)
end

"""
    wavelength_to_wavenumber(λ; output_unit=u"cm^-1")

Convert wavelength to wavenumber. Input assumed in nm if unitless.
"""
function wavelength_to_wavenumber(λ::Real; output_unit=Unitful.cm^-1)
    λ <= 0 && throw(ArgumentError("Wavelength must be positive"))
    λ_cm = λ * 1e-7
    ν̃ = 1 / λ_cm
    return Unitful.uconvert(output_unit, ν̃ * Unitful.cm^-1)
end

function wavelength_to_wavenumber(λ::Unitful.Quantity; output_unit=Unitful.cm^-1)
    λ_cm = Unitful.ustrip(Unitful.uconvert(Unitful.cm, λ))
    λ_cm <= 0 && throw(ArgumentError("Wavelength must be positive"))
    ν̃ = 1 / λ_cm
    return Unitful.uconvert(output_unit, ν̃ * Unitful.cm^-1)
end

# ============================================================================
# Energy conversions
# ============================================================================

"""
    wavenumber_to_energy(ν̃; output_unit=u"eV")

Convert wavenumber (cm⁻¹) to photon energy.
"""
function wavenumber_to_energy(ν̃::Real; output_unit=Unitful.eV)
    ν̃ <= 0 && throw(ArgumentError("Wavenumber must be positive"))
    E_eV = hc_eVcm * ν̃
    return Unitful.uconvert(output_unit, E_eV * Unitful.eV)
end

function wavenumber_to_energy(ν̃::Unitful.Quantity; output_unit=Unitful.eV)
    ν̃_cm = Unitful.ustrip(Unitful.uconvert(Unitful.cm^-1, ν̃))
    return wavenumber_to_energy(ν̃_cm; output_unit)
end

"""
    wavelength_to_energy(λ; output_unit=u"eV")

Convert wavelength to photon energy. Input assumed in nm if unitless.
"""
function wavelength_to_energy(λ::Real; output_unit=Unitful.eV)
    λ <= 0 && throw(ArgumentError("Wavelength must be positive"))
    E_eV = hc_eVnm / λ
    return Unitful.uconvert(output_unit, E_eV * Unitful.eV)
end

function wavelength_to_energy(λ::Unitful.Quantity; output_unit=Unitful.eV)
    λ_nm = Unitful.ustrip(Unitful.uconvert(Unitful.nm, λ))
    return wavelength_to_energy(λ_nm; output_unit)
end

"""
    energy_to_wavelength(E; output_unit=u"nm")

Convert photon energy to wavelength. Input assumed in eV if unitless.
"""
function energy_to_wavelength(E::Real; output_unit=Unitful.nm)
    E <= 0 && throw(ArgumentError("Energy must be positive"))
    λ_nm = hc_eVnm / E
    return Unitful.uconvert(output_unit, λ_nm * Unitful.nm)
end

function energy_to_wavelength(E::Unitful.Quantity; output_unit=Unitful.nm)
    E_eV = Unitful.ustrip(Unitful.uconvert(Unitful.eV, E))
    return energy_to_wavelength(E_eV; output_unit)
end

# ============================================================================
# Linewidth <-> Decay time conversions
# ============================================================================

"""
    decay_time_to_linewidth(τ; output_unit=u"meV")

Convert excited-state decay time to natural linewidth (FWHM): Γ = ℏ/τ.
Input assumed in ps if unitless.
"""
function decay_time_to_linewidth(τ::Real; output_unit=Unitful.meV)
    τ <= 0 && throw(ArgumentError("Decay time must be positive"))
    Γ_meV = ħ_meVps / τ
    Γ = Γ_meV * Unitful.meV
    return _convert_linewidth(Γ, output_unit)
end

function decay_time_to_linewidth(τ::Unitful.Quantity; output_unit=Unitful.meV)
    τ_ps = Unitful.ustrip(Unitful.uconvert(Unitful.ps, τ))
    return decay_time_to_linewidth(τ_ps; output_unit)
end

"""
    linewidth_to_decay_time(Γ; output_unit=u"ps")

Convert spectral linewidth (FWHM) to decay time: τ = ℏ/Γ.
Input assumed in meV if unitless.
"""
function linewidth_to_decay_time(Γ::Real; output_unit=Unitful.ps)
    Γ <= 0 && throw(ArgumentError("Linewidth must be positive"))
    τ_ps = ħ_meVps / Γ
    return Unitful.uconvert(output_unit, τ_ps * Unitful.ps)
end

function linewidth_to_decay_time(Γ::Unitful.Quantity; output_unit=Unitful.ps)
    Γ_meV = _linewidth_to_meV(Γ)
    return linewidth_to_decay_time(Γ_meV; output_unit)
end

# ============================================================================
# Helper functions for linewidth conversions
# ============================================================================

function _linewidth_to_meV(Γ::Unitful.Quantity)
    dim = Unitful.dimension(Γ)

    if dim == Unitful.dimension(u"eV")
        return Unitful.ustrip(Unitful.uconvert(u"meV", Γ))
    elseif dim == Unitful.dimension(u"Hz")
        E = h * Γ
        return Unitful.ustrip(Unitful.uconvert(u"meV", E))
    elseif dim == Unitful.dimension(u"cm^-1")
        ν̃_cm = Unitful.ustrip(Unitful.uconvert(u"cm^-1", Γ))
        E_eV = hc_eVcm * ν̃_cm
        return E_eV * 1000
    else
        throw(ArgumentError("Unsupported linewidth unit. Use energy (eV, meV), frequency (Hz, THz), or wavenumber (cm⁻¹)"))
    end
end

function _convert_linewidth(Γ::Unitful.Quantity, output_unit)
    out_dim = Unitful.dimension(output_unit)

    if out_dim == Unitful.dimension(u"eV")
        return Unitful.uconvert(output_unit, Γ)
    elseif out_dim == Unitful.dimension(u"Hz")
        ν = Γ / h
        return Unitful.uconvert(output_unit, ν)
    elseif out_dim == Unitful.dimension(u"cm^-1")
        Γ_eV = Unitful.ustrip(Unitful.uconvert(u"eV", Γ))
        ν̃ = Γ_eV / hc_eVcm
        return Unitful.uconvert(output_unit, ν̃ * u"cm^-1")
    else
        throw(ArgumentError("Unsupported output unit. Use energy (eV, meV), frequency (Hz, THz), or wavenumber (cm⁻¹)"))
    end
end
