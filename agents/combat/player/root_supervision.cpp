#include "agents/combat/player/root_supervision.hpp"
#include "agents/combat/search/gumbel/gumbel.hpp"
#include <limits>
#include <nlohmann/json.hpp>

namespace stsrl::player {
namespace {
using Json = nlohmann::json;
constexpr double probe_probability = 0.0625;
constexpr search::Gumbel::Settings probe_settings{.simulations = 256, .considered_actions = 16, .particles = 16,
                                                  .spend_even_budget = true};

Json checkpoint(const search::Gumbel::Round& r) {
    return {{"simulations", r.simulations}, {"prior", r.prior.bits}, {"selected", r.selected.bits},
            {"prior_q", r.prior_q ? Json(*r.prior_q) : Json(nullptr)},
            {"selected_q", r.selected_q ? Json(*r.selected_q) : Json(nullptr)}};
}
}  // namespace

struct RootSupervision::Impl {
    std::optional<Decision> probe;
    std::vector<search::Gumbel::Round> rounds;
    std::optional<Labels> labels;
    Json traces = Json::array();
    std::array<int, nn::layout::sample::thinking_rounds> coverage{};
    std::size_t evals = 0, sample = 0;
    int step = 0;
};

RootSupervision::RootSupervision() : impl_{std::make_unique<Impl>()} {}
RootSupervision::~RootSupervision() = default;

RootSupervision::Labels RootSupervision::missing() {
    Labels labels;
    labels.fill(std::numeric_limits<float>::quiet_NaN());
    return labels;
}

void RootSupervision::observe(const sts::BattleContext& state, std::mt19937_64 random,
                              const nn::NetworkEvaluator& network, int step, std::size_t sample) {
    impl_->labels.reset();
    impl_->rounds.clear();
    if (!std::bernoulli_distribution(probe_probability)(random)) return;
    impl_->step = step;
    impl_->sample = sample;
    impl_->probe.emplace(std::make_unique<search::Gumbel>(state, probe_settings, random(),
        [this](const search::Gumbel::Round& round) { impl_->rounds.push_back(round); }), network);
}

std::span<const infra::Tokens> RootSupervision::advance() {
    if (!impl_->probe) return {};
    if (const auto requests = impl_->probe->pending(); !requests.empty()) return requests;
    auto labels = missing();
    Json checkpoints = Json::array();
    for (std::size_t i = 0; i < impl_->rounds.size(); ++i) {
        const auto& round = impl_->rounds[i];
        if (const auto gain = round.gain()) {
            labels.at(i) = *gain;
            ++impl_->coverage.at(i);
        }
        checkpoints.push_back(checkpoint(round));
    }
    impl_->labels = labels;
    impl_->evals += impl_->probe->sent();
    impl_->traces.push_back({{"step", impl_->step}, {"sample", impl_->sample},
                            {"evals", impl_->probe->sent()}, {"rounds", std::move(checkpoints)}});
    impl_->probe.reset();
    return {};
}

bool RootSupervision::active() const { return impl_->probe.has_value(); }
void RootSupervision::evaluated(std::span<const infra::Prediction> predictions) { impl_->probe->evaluated(predictions); }
std::optional<RootSupervision::Labels> RootSupervision::labels() const { return impl_->labels; }

Json RootSupervision::summary() const {
    return {{"evals", impl_->evals}, {"coverage", impl_->coverage}, {"probes", impl_->traces},
        {"settings", {{"probability", probe_probability}, {"simulations", probe_settings.simulations},
            {"particles", probe_settings.particles}, {"considered_actions", probe_settings.considered_actions},
            {"wave", probe_settings.wave}, {"world_draw", probe_settings.world_draw},
            {"gumbel_scale", probe_settings.gumbel_scale}, {"sieve_gain", probe_settings.sieve_gain},
            {"value_scale", probe_settings.value_scale}, {"maxvisit_init", probe_settings.maxvisit_init},
            {"spend_even_budget", probe_settings.spend_even_budget}, {"rollout_mix", 0}}}};
}
}  // namespace stsrl::player
