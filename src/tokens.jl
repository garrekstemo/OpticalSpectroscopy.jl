"""
Metadata token contract: controlled vocabularies, label derivation, instrument-
string normalization, and the Symbol↔Unitful bridge.

See docs/superpowers/2026-06-15-metadata-token-contract.md. Every axis is a
`(quantity, unit)` pair of `Symbol` tokens; display labels are *derived* from
those tokens, never stored as prose and never guessed from data magnitudes.
"""

# Controlled vocabularies (§3). Open for extension, closed for validation:
# an unknown value warns and falls back — it never silently passes.
const TECHNIQUES = Set{Symbol}([:ta, :ftir, :raman, :uvvis, :pl, :xrd])

const AXIS_QUANTITIES = Set{Symbol}([
    :wavelength, :wavenumber, :raman_shift, :opd, :two_theta,
    :time, :energy, :position])

const SIGNAL_QUANTITIES = Set{Symbol}([
    :absorbance, :transmittance, :reflectance, :single_beam,
    :interferogram, :delta_absorbance, :delta_transmittance,
    :intensity, :kubelka_munk])

const UNITS = Set{Symbol}([
    :nm, :um, :angstrom, :per_cm, :fs, :ps, :ns, :degree, :eV, :meV,
    :points, :mm, :counts, :arb, :OD, :mOD, :percent, :fraction, :dimensionless])

const _QUANTITY_DISPLAY = Dict{Symbol,String}(
    :wavelength       => "Wavelength",
    :wavenumber       => "Wavenumber",
    :raman_shift      => "Raman shift",
    :opd              => "Optical path difference",
    :two_theta        => "2θ",
    :time             => "Time",
    :energy           => "Energy",
    :position         => "Position",
    :absorbance       => "Absorbance",
    :transmittance    => "Transmittance",
    :reflectance      => "Reflectance",
    :single_beam      => "Single-beam intensity",
    :interferogram    => "Interferogram",
    :delta_absorbance => "ΔA",
    :delta_transmittance => "−ΔT/T",
    :intensity        => "Intensity",
    :kubelka_munk     => "F(R)",
)

const _UNIT_DISPLAY = Dict{Symbol,String}(
    :nm => "nm", :um => "µm", :angstrom => "Å", :per_cm => "cm⁻¹",
    :fs => "fs", :ps => "ps", :ns => "ns", :degree => "°",
    :eV => "eV", :meV => "meV", :points => "points", :mm => "mm",
    :counts => "counts", :arb => "arb. units", :OD => "OD", :mOD => "mOD",
    :percent => "%", :fraction => "", :dimensionless => "",
)

_token_display(s::Symbol) = uppercasefirst(replace(String(s), '_' => ' '))

"""
    axis_label(quantity::Symbol, unit::Symbol) -> String

Compose a display label from a quantity token and a unit token:
`axis_label(:wavenumber, :per_cm) == "Wavenumber (cm⁻¹)"`. A dimensionless or
empty unit yields just the quantity name:
`axis_label(:delta_absorbance, :dimensionless) == "ΔA"`. Unknown tokens fall
back to their humanized `String` form so nothing is ever lost.
"""
function axis_label(quantity::Symbol, unit::Symbol)
    name = get(_QUANTITY_DISPLAY, quantity, _token_display(quantity))
    usym = get(_UNIT_DISPLAY, unit, String(unit))
    return isempty(usym) ? name : "$name ($usym)"
end

# Dict-based label resolvers (§4 precedence). Pure: they take metadata, never
# the data values. No legacy bridge, no magnitude guess.

"""
    _spectral_xlabel(metadata) -> String

X-axis label by §4 precedence: (1) literal `:xlabel` → (2) token-derived from
`:xquantity`/`:xunit` → (3) honest generic floor `"x"`.
"""
function _spectral_xlabel(metadata::AbstractDict)
    lbl = get(metadata, :xlabel, nothing)
    isnothing(lbl) || return String(lbl)
    q = get(metadata, :xquantity, nothing)
    if !isnothing(q)
        u = Symbol(get(metadata, :xunit, :dimensionless))
        return axis_label(Symbol(q), u)
    end
    return "x"
end

"""
    _signal_label(metadata) -> String

Signal-axis label by §4 precedence: (1) literal `:ylabel` → (2) token-derived
from `:yquantity`/`:yunit` → (3) honest generic floor `"Signal"`.
"""
function _signal_label(metadata::AbstractDict)
    lbl = get(metadata, :ylabel, nothing)
    isnothing(lbl) || return String(lbl)
    q = get(metadata, :yquantity, nothing)
    if !isnothing(q)
        u = Symbol(get(metadata, :yunit, :dimensionless))
        return axis_label(Symbol(q), u)
    end
    return "Signal"
end

"""
    _time_label(metadata, unit_key) -> String

Time-axis label: `axis_label(:time, unit)` where `unit` is read from `unit_key`
(`:xunit` for `KineticTrace`, `:time_unit` for `TimeResolvedMatrix`), defaulting
to `:ps`.
"""
_time_label(metadata::AbstractDict, unit_key::Symbol) =
    axis_label(:time, Symbol(get(metadata, unit_key, :ps)))

# Validation (§8). Warns, never throws — out-of-vocabulary values are allowed.

"""
    is_canonical(slot::Symbol, value::Symbol) -> Bool

Whether `value` is in the controlled vocabulary for `slot`. Slots: `:technique`,
`:quantity` (axis or signal), `:unit`. Unknown slots return `false`.
"""
function is_canonical(slot::Symbol, value::Symbol)
    slot === :technique && return value in TECHNIQUES
    slot === :quantity  && return value in AXIS_QUANTITIES || value in SIGNAL_QUANTITIES
    slot === :unit      && return value in UNITS
    return false
end

const _RESERVED_TOKEN_SLOTS = (
    (:technique, :technique),
    (:xquantity, :quantity), (:yquantity, :quantity), (:signal_quantity, :quantity),
    (:axis1_quantity, :quantity), (:axis2_quantity, :quantity), (:axis3_quantity, :quantity),
    (:xunit, :unit), (:yunit, :unit), (:signal_unit, :unit),
    (:axis1_unit, :unit), (:axis2_unit, :unit), (:axis3_unit, :unit),
    (:time_unit, :unit), (:time_delay_unit, :unit), (:gate_unit, :unit),
    (:fixed_coordinate_unit, :unit), (:position_unit, :unit),
)

"""
    validate_tokens(metadata::AbstractDict) -> Bool

Check reserved token values against the controlled vocabularies of §3. `@warn`s
for each reserved key whose value is outside its vocabulary and returns `false`;
returns `true` when all present reserved tokens are canonical. Never throws.
"""
function validate_tokens(metadata::AbstractDict)
    ok = true
    for (key, slot) in _RESERVED_TOKEN_SLOTS
        haskey(metadata, key) || continue
        value = metadata[key]
        value isa Symbol || (value = Symbol(value))
        if !is_canonical(slot, value)
            @warn "Non-canonical metadata token" key value slot
            ok = false
        end
    end
    return ok
end

# Instrument-string normalization (§8). Owned here so every loader maps strings
# the same way.

const _UNIT_ALIASES = Dict{String,Symbol}(
    "nm" => :nm, "nanometers" => :nm, "nanometer" => :nm,
    "um" => :um, "µm" => :um, "μm" => :um, "micron" => :um, "microns" => :um, "micrometers" => :um,
    "angstrom" => :angstrom, "angstroms" => :angstrom, "å" => :angstrom,
    "cm-1" => :per_cm, "cm^-1" => :per_cm, "cm⁻¹" => :per_cm, "1/cm" => :per_cm, "per cm" => :per_cm,
    "fs" => :fs, "ps" => :ps, "ns" => :ns,
    "deg" => :degree, "degree" => :degree, "degrees" => :degree, "°" => :degree,
    "ev" => :eV, "mev" => :meV,
    "points" => :points, "point" => :points, "pts" => :points,
    "mm" => :mm,
    "counts" => :counts, "count" => :counts, "cts" => :counts,
    "arb" => :arb, "arbitrary" => :arb, "a.u." => :arb, "au" => :arb,
    "od" => :OD, "mod" => :mOD,
    "%" => :percent, "%t" => :percent, "%r" => :percent, "percent" => :percent,
    "fraction" => :fraction,
    "dimensionless" => :dimensionless, "" => :dimensionless,
)

const _QUANTITY_ALIASES = Dict{String,Symbol}(
    "absorbance" => :absorbance, "abs" => :absorbance,
    "transmittance" => :transmittance, "%t" => :transmittance, "transmission" => :transmittance,
    "reflectance" => :reflectance, "%r" => :reflectance, "reflection" => :reflectance,
    "single beam" => :single_beam, "single-beam" => :single_beam, "sb" => :single_beam,
    "interferogram" => :interferogram, "interferrogram" => :interferogram, "igram" => :interferogram,
    "delta absorbance" => :delta_absorbance, "δa" => :delta_absorbance, "deltaa" => :delta_absorbance,
    "intensity" => :intensity, "counts" => :intensity, "pl" => :intensity,
    "wavelength" => :wavelength, "wavenumber" => :wavenumber,
    "raman shift" => :raman_shift, "raman" => :raman_shift,
    "2theta" => :two_theta, "two theta" => :two_theta, "2θ" => :two_theta,
    "time" => :time, "delay" => :time, "energy" => :energy,
)

_fallback_token(key::AbstractString) = Symbol(replace(key, ' ' => '_'))

"""
    normalize_unit(s::AbstractString) -> Symbol

Map a free-form instrument unit string to a canonical unit token (`"NANOMETERS"
→ :nm`, `"1/cm" → :per_cm`). Unrecognized strings return `Symbol` of the
lowercased, underscored text so nothing is lost.
"""
normalize_unit(s::AbstractString) =
    get(_UNIT_ALIASES, lowercase(strip(s)), _fallback_token(lowercase(strip(s))))

"""
    normalize_quantity(s::AbstractString) -> Symbol

Map a free-form instrument quantity string to a canonical quantity token
(`"ABSORBANCE" → :absorbance`, `"Raman Shift" → :raman_shift`). Unrecognized
strings return `Symbol` of the lowercased, underscored text.
"""
normalize_quantity(s::AbstractString) =
    get(_QUANTITY_ALIASES, lowercase(strip(s)), _fallback_token(lowercase(strip(s))))

# Symbol↔Unitful bridge (§6) and canonical unit per quantity.

const _UNITFUL_MAP = Dict{Symbol,Any}(
    :nm => u"nm", :um => u"µm", :angstrom => u"angstrom", :per_cm => u"cm^-1",
    :fs => u"fs", :ps => u"ps", :ns => u"ns", :degree => u"°",
    :eV => u"eV", :meV => u"meV", :mm => u"mm",
)

"""
    _unitful(unit::Symbol)

Map a unit token to its Unitful unit (`_unitful(:per_cm) === u"cm^-1"`). Tokens
Unitful cannot represent (`:OD`, `:counts`, `:arb`, `:points`, `:percent`, …)
map to `Unitful.NoUnits`, the identity under `*`.
"""
_unitful(unit::Symbol) = get(_UNITFUL_MAP, unit, Unitful.NoUnits)

"""
    CANONICAL_UNIT

Default unit token per quantity token, used by the `axis=` constructor sugar.
"""
const CANONICAL_UNIT = Dict{Symbol,Symbol}(
    :wavelength => :nm, :wavenumber => :per_cm, :raman_shift => :per_cm,
    :opd => :points, :two_theta => :degree, :time => :ps, :energy => :eV,
    :position => :um,
    :absorbance => :OD, :transmittance => :percent, :reflectance => :percent,
    :single_beam => :arb, :interferogram => :arb, :delta_absorbance => :mOD,
    :intensity => :counts,
)
