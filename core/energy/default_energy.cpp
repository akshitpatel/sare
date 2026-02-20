#include "energy/default_energy.hpp"
#include "energy/syntax_energy.hpp"
#include "energy/complexity_energy.hpp"
#include "energy/resource_energy.hpp"
#include "energy/constraint_energy.hpp"
#include "energy/test_energy.hpp"

namespace sare {

EnergyAggregator makeDefaultEnergyAggregator(const EnergyWeights& weights) {
    EnergyAggregator energy(weights);
    energy.addComponent(std::make_unique<SyntaxEnergy>());
    energy.addComponent(std::make_unique<ConstraintEnergy>());
    energy.addComponent(std::make_unique<TestEnergy>());
    energy.addComponent(std::make_unique<ComplexityEnergy>());
    energy.addComponent(std::make_unique<ResourceEnergy>());
    energy.addComponent(std::make_unique<UncertaintyEnergy>());
    return energy;
}

} // namespace sare
