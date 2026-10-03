// Parquet writer for PV training rows (one row per decision). Layout documented in agents/combat/pv/data.py.
#pragma once
#include "agents/combat/pv/features.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace stsrl::pv {

struct Row {
    std::string fight_id;
    std::uint64_t seed;
    int encounter, step, turn;
    bool won;
    int final_hp;
    float value_target;
    std::optional<float> root_value;
    std::vector<std::uint32_t> moves;
    std::vector<float> policy_target;  // normalized visits; all zeros when has_policy is false
    bool has_policy;
    const Inputs* inputs;
};

class ShardWriter {
public:
    explicit ShardWriter(const std::string& path) {
        std::vector<std::shared_ptr<arrow::Field>> fields{
            arrow::field("fight_id", arrow::utf8()), arrow::field("seed", arrow::uint64()),
            arrow::field("encounter", arrow::int8()), arrow::field("step", arrow::int32()),
            arrow::field("turn", arrow::int16()), arrow::field("won", arrow::boolean()),
            arrow::field("final_hp", arrow::int16()), arrow::field("value_target", arrow::float32()),
            arrow::field("root_value", arrow::float32()), arrow::field("moves", arrow::list(arrow::uint32())),
            arrow::field("policy_target", arrow::list(arrow::float32())), arrow::field("has_policy", arrow::boolean())};
        for (std::size_t i = 0; i < names.size(); ++i) fields.push_back(arrow::field(names[i], arrow::list(arrow::float32())));
        for (std::size_t i = 1; i < names.size(); ++i) fields.push_back(arrow::field(std::string{"n_"} + names[i], arrow::int16()));
        schema_ = arrow::schema(fields, arrow::key_value_metadata({"pv_contract"}, {contract}));
        out_ = value(arrow::io::FileOutputStream::Open(path));
        auto props = parquet::WriterProperties::Builder().compression(parquet::Compression::ZSTD)->build();
        auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();
        writer_ = value(parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), out_, props, arrow_props));
    }

    void add(const Row& r) {
        ok(fight_id_.Append(r.fight_id)); ok(seed_.Append(r.seed)); ok(encounter_.Append(r.encounter));
        ok(step_.Append(r.step)); ok(turn_.Append(r.turn)); ok(won_.Append(r.won)); ok(hp_.Append(r.final_hp));
        ok(target_.Append(r.value_target));
        if (r.root_value) ok(root_.Append(*r.root_value)); else ok(root_.AppendNull());
        moves_.add(r.moves.data(), r.moves.size());
        policy_.add(r.policy_target.data(), r.policy_target.size());
        ok(has_policy_.Append(r.has_policy));
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto& x = (*r.inputs)[i];
            features_[i].add(x.data(), x.size());
            if (i) ok(counts_[i - 1].Append(std::int16_t(x.size() / widths[i])));
        }
        if (++pending_ == 1000) flush();
    }

    void close() {
        flush();
        ok(writer_->Close());
        ok(out_->Close());
    }

private:
    static void ok(const arrow::Status& s) { if (!s.ok()) throw std::runtime_error{s.ToString()}; }
    template <class T> static T value(arrow::Result<T> r) { ok(r.status()); return std::move(*r); }

    template <class B> struct ListColumn {
        B* values;
        std::shared_ptr<arrow::ListBuilder> list;
        ListColumn() {
            auto v = std::make_shared<B>();
            values = v.get();
            list = std::make_shared<arrow::ListBuilder>(arrow::default_memory_pool(), v);
        }
        template <class T> void add(const T* data, std::size_t n) { ok(list->Append()); ok(values->AppendValues(data, n)); }
    };

    void flush() {
        if (!pending_) return;
        std::vector<std::shared_ptr<arrow::Array>> columns;
        auto finish = [&](auto& builder) { columns.push_back(value(builder.Finish())); };
        finish(fight_id_); finish(seed_); finish(encounter_); finish(step_); finish(turn_); finish(won_); finish(hp_);
        finish(target_); finish(root_); finish(*moves_.list); finish(*policy_.list); finish(has_policy_);
        for (auto& f : features_) finish(*f.list);
        for (auto& c : counts_) finish(c);
        ok(writer_->WriteTable(*arrow::Table::Make(schema_, columns), 1000));
        pending_ = 0;
    }

    std::shared_ptr<arrow::Schema> schema_;
    std::shared_ptr<arrow::io::FileOutputStream> out_;
    std::unique_ptr<parquet::arrow::FileWriter> writer_;
    int pending_ = 0;
    arrow::StringBuilder fight_id_;
    arrow::UInt64Builder seed_;
    arrow::Int8Builder encounter_;
    arrow::Int32Builder step_;
    arrow::Int16Builder turn_, hp_;
    arrow::BooleanBuilder won_, has_policy_;
    arrow::FloatBuilder target_, root_;
    ListColumn<arrow::UInt32Builder> moves_;
    ListColumn<arrow::FloatBuilder> policy_;
    std::array<ListColumn<arrow::FloatBuilder>, 6> features_;
    std::array<arrow::Int16Builder, 5> counts_;
};
}  // namespace stsrl::pv
