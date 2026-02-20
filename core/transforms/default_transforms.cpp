#include "transforms/default_transforms.hpp"
#include "transforms/algebra_transforms.hpp"
#include "transforms/logic_transforms.hpp"
#include "transforms/ast_transforms.hpp"

namespace sare {

void registerDefaultTransforms(TransformRegistry& registry) {
    registry.registerTransform(std::make_unique<AddZeroTransform>());
    registry.registerTransform(std::make_unique<MulOneTransform>());
    registry.registerTransform(std::make_unique<MulZeroTransform>());
    registry.registerTransform(std::make_unique<DoubleNegationTransform>());
    registry.registerTransform(std::make_unique<AndTrueTransform>());
    registry.registerTransform(std::make_unique<ConstantFoldTransform>());
}

} // namespace sare
