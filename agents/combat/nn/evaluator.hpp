#pragma once
// The network as a search evaluator. Asynchronous by design: a query goes out as tokens, and the prediction that
// comes back (from whoever runs inference, e.g. infra's batched backends) is read as the search's evaluation.
#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/search/search.hpp"
#include "infra/inference/backend.hpp"

namespace stsrl::nn {
class NetworkEvaluator {
public:
    explicit NetworkEvaluator(Encoding encoding) : encoding_{encoding} {}
    infra::Tokens request(const search::Query& query) const { return encode(query.state, query.actions, encoding_); }
    static search::Evaluation answer(const infra::Prediction& prediction) { return {prediction.value, prediction.logits}; }

private:
    Encoding encoding_;
};
}  // namespace stsrl::nn
