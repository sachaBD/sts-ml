#pragma once
#include "agents/combat/search/search.hpp"

namespace stsrl::search {
enum class RootChoice { sample, argmax };
SearchFactory sift_sieve(RootChoice);
}  // namespace stsrl::search
