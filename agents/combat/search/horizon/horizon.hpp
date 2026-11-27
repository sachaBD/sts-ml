#pragma once
// Unified Horizon Combat Tree Search.
// One public-observation DAG, common-horizon mid-turn plunging with progressive two-turn boundary
// evaluation, additive world-adjusted root Q over shared CRN worlds, and unified Laplace Max-VOI
// stopping controlled by a single compute-price knob (lambda).
#include "agents/combat/search/search.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace stsrl::search {
class Horizon final : public CombatSearch {
public:
    struct Settings {
        int simulations = 256;       // max root simulations (budget cap)
        int particles = 16;          // particle world pool size cycled with CRN
        int init_contenders = 3;     // initial top-logit contenders plunged together on Wave 1
        int init_worlds = 2;         // shared CRN worlds plunged on Wave 1
        int wave_worlds = 4;         // shared CRN worlds per subsequent paired wave
        int max_plunge_choices = 8;  // non-forced choice cap per mid-turn plunge
        int horizon_turns = 2;       // 1 = Turn+1 boundary; >= 2 = Wave 1 Turn+1, Waves 2+ mean(V1..VH)
        double c_puct = 0.25;        // interior Turn-0 PUCT exploration constant
        double readout_k0 = 30.0;    // prior-strength denominator in Q + lgap / (k0 + visits)
        double prior_se = 0.05;      // Bernoulli-scaled gap uncertainty regularization at low sample counts
        double lambda = 6e-5;        // expected win-rate gain per NN eval required for another wave (0 = fixed cap)
        std::uint64_t world_draw = 0;
    };

    using Wave = Checkpoint;
    using OnWave = std::function<void(const Wave&)>;
    Horizon(const sts::BattleContext& observed, const Settings&, std::uint64_t seed, OnWave = {});
    ~Horizon() override;
    std::span<const Query> pending() const override;
    void evaluated(std::span<const Evaluation>) override;
    Result result() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace stsrl::search
