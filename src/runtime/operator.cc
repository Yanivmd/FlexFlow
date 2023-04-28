#include "flexflow/operator.h"
#include "flexflow/simulator.h"
#include <iomanip>

namespace FlexFlow {

size_t Op::get_untyped_params_hash() const {
  size_t hash = this->get_params_hash();
  hash_combine(hash, this->op_type);
  return hash;
}

size_t Op::get_params_hash() const {
  assert (false);
}

size_t Op::hash() const {
  size_t total_hash = 0;
  hash_combine(total_hash, static_cast<int>(op_type));
  // hash_combine(total_hash, static_cast<int>(data_type));
  for (int i = 0; name[i] != '\0' && name[i] != '_' && i < MAX_OPNAME; i++) {
    hash_combine(total_hash, name[i]);
  }
  for (int i = 0; i < numInputs; i++) {
    hash_combine(total_hash, inputs[i]->hash());
    hash_combine(total_hash, trainableInputs[i]);
  }
  for (int i = 0; i < numOutputs; i++) {
    hash_combine(total_hash, outputs[i]->hash());
  }
  for (int i = 0; i < numWeights; i++) {
    hash_combine(total_hash, weights[i]->hash());
  }
  return total_hash;
}

std::ostream& operator<<(std::ostream& os, const Op& op) {
  int w = 25;
  os << std::setw(w) << "ptr: "                    << &op              << std::endl
     << std::setw(w) << "op_guid: "                << op.op_guid       << std::endl
     << std::setw(w) << "name: "                   << op.name          << std::endl;

  os << std::setw(w) << "outputs: "                                    << std::endl;
  for (int i = 0; i < op.numOutputs; i++)       os << *(op.outputs[i]) << std::endl;

  os << std::setw(w) << "inputs: "                                     << std::endl;
  for (int i = 0; i < op.numInputs ; i++)       os << *(op.inputs[i])  << std::endl;

  os << std::setw(w) << "weights: "                                    << std::endl;
  for (int i = 0; i < op.numWeights ; i++)      os << *(op.weights[i]) << std::endl;

  os << std::setw(w) << "parallel_dims_mapping: "  << op.parallel_dims_mapping << std::endl;
  return os;
}
}; // namespace FlexFlow