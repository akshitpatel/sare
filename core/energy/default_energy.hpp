#pragma once

#include "energy/energy.hpp"

namespace sare {

EnergyAggregator makeDefaultEnergyAggregator(const EnergyWeights& weights = {});

} // namespace sare
