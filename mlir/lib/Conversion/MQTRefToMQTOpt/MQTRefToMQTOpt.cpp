/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any ref gate operation to the same
// gate operation in the opt dialect
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#define ADD_CONVERT_PATTERN(gate)                                              \
  patterns                                                                     \
      .add<ConvertMQTRefGateOp<::mqt::ir::ref::gate, ::mqt::ir::opt::gate>>(   \
          typeConverter, context, &state);

#include "mlir/Conversion/MQTRefToMQTOpt/MQTRefToMQTOpt.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>
#include <vector>

namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_MQTREFTOMQTOPT
#include "mlir/Conversion/MQTRefToMQTOpt/MQTRefToMQTOpt.h.inc"

namespace {

struct LoweringState {
  /// @brief Map each initial ref qubit to its latest opt qubit.
  llvm::DenseMap<Value, Value> qubitMap;
  /// @brief Map each initial ref qubit to its index.
  llvm::DenseMap<Value, Value> qubitIndexMap;
  /// @brief Map each initial ref register to its refQubits.
  llvm::DenseMap<Value, std::vector<Value>> qregQubitsMap;
  /// @brief Map each initial funcOp to its refQubits.
  llvm::DenseMap<func::FuncOp, std::vector<Value>> funcQubitsMap;
  /// @brief Map each initial op to its refQubits.
  llvm::DenseMap<Operation*, llvm::SetVector<Value>> regionMap;
  /// @brief Map each initial funcOp to its refQubits.
  llvm::DenseMap<Region*, llvm::DenseMap<Value, Value>> regionQubitMap;
};

template <typename OpType>
class StatefulOpConversionPattern : public mlir::OpConversionPattern<OpType> {
  using mlir::OpConversionPattern<OpType>::OpConversionPattern;

public:
  StatefulOpConversionPattern(mlir::TypeConverter& typeConverter,
                              mlir::MLIRContext* context, LoweringState* state)
      : mlir::OpConversionPattern<OpType>(typeConverter, context),
        state_(state) {}

  /// @brief Return the state object as reference.
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

bool isQubitType(const MemRefType type) {
  return llvm::isa<ref::QubitType>(type.getElementType());
}

bool isQubitType(memref::AllocOp op) { return isQubitType(op.getType()); }

bool isQubitType(memref::DeallocOp op) {
  const auto& memRef = op.getMemref();
  const auto& memRefType = llvm::cast<MemRefType>(memRef.getType());
  return isQubitType(memRefType);
}

bool isQubitType(memref::LoadOp op) {
  const auto& memRef = op.getMemref();
  const auto& memRefType = llvm::cast<MemRefType>(memRef.getType());
  return isQubitType(memRefType);
}

llvm::SetVector<Value> collectRegionQubits(Operation* op, LoweringState* state,
                                           MLIRContext* ctx) {

  // get the regions of the current operation
  auto regions = op->getRegions();
  SetVector<Value> uniqueQubits;
  for (auto& region : regions) {

    // skip empty regions e.g. empty else region of an If operation
    if (region.empty()) {
      continue;
    }

    for (auto& operation : region.front().getOperations()) {
      // check if the operation has an region, if yes recursively collect the
      // qubits
      if (operation.getNumRegions() > 0) {
        auto qubits = collectRegionQubits(&operation, state, ctx);
        for (auto qubit : qubits) {
          uniqueQubits.insert(qubit);
        }
      }
      // collect qubits form the operands
      for (auto operand : operation.getOperands()) {
        if (operand.getType() == ref::QubitType::get(ctx)) {
          uniqueQubits.insert(operand);
        }
      }
      // collect qubits from the results
      for (auto result : operation.getResults()) {
        if (result.getType() == ref::QubitType::get(ctx)) {
          uniqueQubits.insert(result);
        }
      }
    }
  }
  if (!uniqueQubits.empty() &&
      (llvm::isa<scf::IfOp>(op) || (llvm::isa<scf::ForOp>(op)))) {
    state->regionMap[op] = uniqueQubits;
    op->setAttr("needChange", StringAttr::get(ctx, "yes"));
  }
  return uniqueQubits;
}

} // namespace

class MQTRefToMQTOptTypeConverter final : public TypeConverter {
public:
  explicit MQTRefToMQTOptTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](ref::QubitType /*type*/) -> Type {
      return opt::QubitType::get(ctx);
    });

    // MemRefType conversion
    addConversion([ctx](MemRefType type) -> Type {
      if (isQubitType(type)) {
        return MemRefType::get(type.getShape(), opt::QubitType::get(ctx));
      }
      return type;
    });
  }
};

struct ConvertMQTRefMemRefAlloc final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern<
      memref::AllocOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    const auto& qubitType = opt::QubitType::get(rewriter.getContext());
    auto memRefType = MemRefType::get(op.getType().getShape(), qubitType);

    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memRefType,
                                                 adaptor.getDynamicSizes());

    return success();
  }
};

struct ConvertMQTRefMemRefDealloc final
    : StatefulOpConversionPattern<memref::DeallocOp> {
  using StatefulOpConversionPattern<
      memref::DeallocOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    auto refMemRef = op.getMemref();
    const auto& optMemRef = adaptor.getMemref();

    // iterate over all the qubits that were extracted from the register
    for (const auto& refQubit : getState().qregQubitsMap[refMemRef]) {
      const auto& optQubit = getState().qubitMap[refQubit];
      auto index = getState().qubitIndexMap[refQubit];

      auto storeOp = rewriter.create<memref::StoreOp>(
          op.getLoc(), optQubit, optMemRef, ValueRange{index});

      // move it before the current dealloc operation
      storeOp->moveBefore(op);

      // erase the refQubit entry from the maps
      getState().qubitMap.erase(refQubit);
      getState().qubitIndexMap.erase(refQubit);
    }

    // erase the register from the map
    getState().qregQubitsMap.erase(refMemRef);

    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, optMemRef);

    return success();
  }
};

struct ConvertMQTRefAllocQubit final
    : StatefulOpConversionPattern<ref::AllocQubitOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::AllocQubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& refQubit = op.getQubit();
    auto optOp = rewriter.replaceOpWithNewOp<opt::AllocQubitOp>(op);
    const auto& optQubit = optOp.getQubit();
    getState().qubitMap.try_emplace(refQubit, optQubit);
    auto* regionOp = op->getParentRegion();
    getState().regionQubitMap[regionOp].try_emplace(refQubit, optQubit);
    return success();
  }
};

struct ConvertMQTRefDeallocQubit final
    : StatefulOpConversionPattern<ref::DeallocQubitOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::DeallocQubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& refQubit = op.getQubit();
    const auto& optQubit = getState().qubitMap[refQubit];
    rewriter.replaceOpWithNewOp<opt::DeallocQubitOp>(op, optQubit);
    getState().qubitMap.erase(refQubit);
    return success();
  }
};

struct ConvertMQTRefMemRefLoad final
    : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern<
      memref::LoadOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    // create new operation
    auto optLoadOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, adaptor.getMemref(), adaptor.getIndices());

    const auto& refMemRef = op.getMemref();
    const auto& refQubit = op.getResult();
    const auto& optQubit = optLoadOp.getResult();

    auto* regionOp = op->getParentRegion();
    getState().regionQubitMap[regionOp].try_emplace(refQubit, optQubit);

    // put the pair of the ref qubit and the latest opt qubit in the map
    getState().qubitMap.try_emplace(refQubit, optQubit);

    // add entry to qubitIndexMap
    getState().qubitIndexMap.try_emplace(refQubit,
                                         adaptor.getIndices().front());

    // append the entry to the qregQubitsMap to store which qubits that were
    // extracted from the register
    getState().qregQubitsMap[refMemRef].emplace_back(refQubit);

    return success();
  }
};

struct ConvertMQTRefMeasure final
    : StatefulOpConversionPattern<ref::MeasureOp> {
  using StatefulOpConversionPattern<
      ref::MeasureOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::MeasureOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // prepare result type
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    const auto& refQubit = op.getInQubit();

    // get the latest opt qubit from the map and add them to the vector
    const Value optQubit = getState().qubitMap[refQubit];

    // create new operation
    auto optOp = rewriter.create<opt::MeasureOp>(
        op.getLoc(), qubitType, op.getOutBit().getType(), optQubit);

    auto outOptQubit = optOp.getOutQubit();
    auto newBit = optOp.getOutBit();

    getState().qubitMap[refQubit] = outOptQubit;
    auto* regionOp = op->getParentRegion();
    getState().regionQubitMap[regionOp][refQubit] = outOptQubit;
    // replace the old operation results with the new bits and delete
    // old operation
    rewriter.replaceOp(op, newBit);

    return success();
  }
};

struct ConvertMQTRefReset final : StatefulOpConversionPattern<ref::ResetOp> {
  using StatefulOpConversionPattern<ref::ResetOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::ResetOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    // prepare result type
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    const auto& refQubit = op.getInQubit();

    // get the latest opt qubit from the map and add them to the vector
    const Value optQubit = getState().qubitMap[refQubit];

    // create new operation
    auto optOp =
        rewriter.create<opt::ResetOp>(op.getLoc(), qubitType, optQubit);

    auto outOptQubit = optOp.getOutQubit();

    getState().qubitMap[refQubit] = outOptQubit;

    // delete the old operation
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertMQTRefQubit final : StatefulOpConversionPattern<ref::QubitOp> {
  using StatefulOpConversionPattern<ref::QubitOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(ref::QubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // prepare result type
    const auto& qubitType = opt::QubitType::get(rewriter.getContext());

    // create new operation
    auto optOp =
        rewriter.create<opt::QubitOp>(op.getLoc(), qubitType, op.getIndex());

    // collect ref and opt SSA value
    const auto& refQubit = op.getQubit();
    const auto& optQubit = optOp.getQubit();

    // map ref to opt
    getState().qubitMap[refQubit] = optQubit;

    // replace the old operation result with the new result and delete
    // old operation
    rewriter.replaceOp(op, optQubit);

    return success();
  }
};

template <typename MQTGateRefOp, typename MQTGateOptOp>
struct ConvertMQTRefGateOp final : StatefulOpConversionPattern<MQTGateRefOp> {
  using StatefulOpConversionPattern<MQTGateRefOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateRefOp op, typename MQTGateRefOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto region = op->getParentRegion();
    auto maps = this->getState().regionQubitMap[region];
    auto mapQubits = [&](const auto& refQubits) {
      std::vector<Value> optQubits;
      for (const auto& refQubit : refQubits) {
        optQubits.emplace_back(
            this->getState().regionQubitMap[region][refQubit]);
      }
      return optQubits;
    };

    auto optInQubits = mapQubits(op.getInQubits());
    auto optPosCtrlQubitsValues = mapQubits(op.getPosCtrlInQubits());
    auto optNegCtrlQubitsValues = mapQubits(op.getNegCtrlInQubits());

    // Get optional attributes
    auto staticParams = op.getStaticParams()
                            ? DenseF64ArrayAttr::get(rewriter.getContext(),
                                                     *op.getStaticParams())
                            : DenseF64ArrayAttr{};
    auto paramMask = op.getParamsMask()
                         ? DenseBoolArrayAttr::get(rewriter.getContext(),
                                                   *op.getParamsMask())
                         : DenseBoolArrayAttr{};

    // Create new operation
    auto optOp = rewriter.create<MQTGateOptOp>(
        op.getLoc(), ValueRange(optInQubits).getTypes(),
        ValueRange(optPosCtrlQubitsValues).getTypes(),
        ValueRange(optNegCtrlQubitsValues).getTypes(), staticParams, paramMask,
        op.getParams(), optInQubits, optPosCtrlQubitsValues,
        optNegCtrlQubitsValues);

    // Update qubit map
    const auto& optResults = optOp.getAllOutQubits();
    for (size_t i = 0; i < op.getAllInQubits().size(); i++) {
      this->getState().regionQubitMap[region][op.getAllInQubits()[i]] =
          optResults[i];
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCallOpOpt final : StatefulOpConversionPattern<func::CallOp> {
  using StatefulOpConversionPattern<func::CallOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;

    auto const inputs = op.getOperands();
    SmallVector<Value> params;
    SmallVector<Value> refQubits;
    auto const optType = opt::QubitType::get(rewriter.getContext());
    for (auto value : inputs) {
      if (isa<ref::QubitType>(value.getType())) {
        resultTypes.push_back(optType);
        params.push_back(this->getState().qubitMap[value]);
        refQubits.push_back(value);
      } else {
        resultTypes.push_back(value.getType());
        params.push_back(value);
      }
    }

    auto callOp = rewriter.create<func::CallOp>(
        op->getLoc(), adaptor.getCallee(), resultTypes, params);

    auto callResults = callOp->getResults();
    for (auto refQubit : refQubits) {
      while (!callResults.empty() &&
             !isa<opt::QubitType>((*callResults.begin()).getType())) {
        //       llvm::outs()<<"here2";
        callResults.drop_front();
      }
      //  llvm::outs()<<"here";
      this->getState().qubitMap[refQubit] = *callResults.begin();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertIfOpOpt final : StatefulOpConversionPattern<scf::IfOp> {
  using StatefulOpConversionPattern<scf::IfOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto refQubits = getState().regionMap[op];
    SmallVector<Value> values;
    values.reserve(refQubits.size());
    for (auto qubit : refQubits) {
      values.push_back(qubit);
    }
    auto const optType = opt::QubitType::get(rewriter.getContext());
    SmallVector<Type> resultTypes;
    resultTypes.assign(refQubits.size(), optType);

    auto newIf = rewriter.create<scf::IfOp>(
        op->getLoc(), TypeRange{resultTypes}, op.getCondition(), true);

    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().end());
    rewriter.eraseBlock(&newIf.getThenRegion().front());

    if (!op.getElseRegion().empty()) {
      rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(),
                                  newIf.getElseRegion().end());
      rewriter.eraseBlock(&newIf.getElseRegion().front());
    }

    auto& thenRegion = newIf.getThenRegion();
    auto& elseRegion = newIf.getElseRegion();

    Operation* thenYield = nullptr;
    Operation* elseYield = nullptr;

    rewriter.setInsertionPointToEnd(&thenRegion.front());
    if (thenRegion.front().getTerminator() == nullptr) {
      thenYield = rewriter.create<scf::YieldOp>(op->getLoc(), values);
    } else {
      thenYield = rewriter.replaceOpWithNewOp<scf::YieldOp>(
          thenRegion.front().getTerminator(), values);
    }
    rewriter.setInsertionPointToEnd(&elseRegion.front());
    if (elseRegion.front().empty()) {
      elseYield = rewriter.create<scf::YieldOp>(op->getLoc(), values);
    } else {
      elseYield = rewriter.replaceOpWithNewOp<scf::YieldOp>(
          elseRegion.front().getTerminator(), values);
    }

    rewriter.setInsertionPoint(op);

    thenYield->setAttr("needChange", rewriter.getStringAttr("yes"));
    elseYield->setAttr("needChange", rewriter.getStringAttr("yes"));
    auto& thenRegionQubitMap = getState().regionQubitMap[&thenRegion];
    auto& elseRegionQubitMap = getState().regionQubitMap[&elseRegion];
    for (const auto& refQubit : refQubits) {
      thenRegionQubitMap.try_emplace(
          refQubit, getState().regionQubitMap[op->getParentRegion()][refQubit]);
      elseRegionQubitMap.try_emplace(
          refQubit, getState().regionQubitMap[op->getParentRegion()][refQubit]);
    }
    auto& qubitMap = getState().regionQubitMap[op->getParentRegion()];
    for (size_t i = 0; i < newIf->getResults().size(); i++) {
      qubitMap[refQubits[i]] = newIf->getResult(i);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertFuncOpOpt final : StatefulOpConversionPattern<func::FuncOp> {
  using StatefulOpConversionPattern<func::FuncOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    SmallVector<Type> argumentTypes;
    auto optType = opt::QubitType::get(rewriter.getContext());
    SmallVector<Type> resultTypes;
    SmallVector<Value> refQubits;
    for (auto type : op->getResultTypes()) {
      if (!isa<ref::QubitType>(type)) {
        resultTypes.push_back(type);
      }
    }
    for (auto blockArg : op.front().getArguments()) {
      if (isa<ref::QubitType>(blockArg.getType())) {
        refQubits.emplace_back(blockArg);
        getState().funcQubitsMap[op].emplace_back(blockArg);
        blockArg.setType(optType);
        getState().qubitMap[refQubits[0]] = blockArg;
        argumentTypes.push_back(optType);
        resultTypes.push_back(optType);

      } else {
        argumentTypes.push_back(blockArg.getType());
      }
    }

    auto newFuncType = rewriter.getFunctionType(argumentTypes, resultTypes); //
    op.setFunctionType(newFuncType);

    op.walk([&](func::ReturnOp returnOp) {
      returnOp->setAttr("needChange", rewriter.getStringAttr("yes"));
    });
    return success();
  }
};

struct ConvertReturnOpOpt final : StatefulOpConversionPattern<func::ReturnOp> {
  using StatefulOpConversionPattern<
      func::ReturnOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    auto const parentFunc = op->getParentOfType<func::FuncOp>();
    SmallVector<Value> results;
    results.append(op->getResults().begin(), op->getResults().end());
    for (auto refQubit : this->getState().funcQubitsMap[parentFunc]) {
      results.emplace_back(getState().qubitMap[refQubit]);
    }

    rewriter.create<func::ReturnOp>(op->getLoc(), results);

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertYieldOpOpt final : StatefulOpConversionPattern<scf::YieldOp> {
  using StatefulOpConversionPattern<scf::YieldOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    auto* region = op->getParentRegion();
    auto& qubitMap = getState().regionQubitMap[region];

    SmallVector<Value> optQubits;
    for (auto refQubit : op->getOperands()) {
      if (refQubit.getType() == ref::QubitType::get(rewriter.getContext())) {
        optQubits.push_back(qubitMap[refQubit]);
      }
    }
    auto yieldOp = rewriter.replaceOpWithNewOp<scf::YieldOp>(op, optQubits);
    yieldOp->setAttr("moreChange", rewriter.getStringAttr("yes"));
    return success();
  }
};
struct ConvertYieldOpOpt2 final : StatefulOpConversionPattern<scf::YieldOp> {
  using StatefulOpConversionPattern<scf::YieldOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    auto* region = op->getParentRegion();
    auto& qubitMap = getState().regionQubitMap[region];

    SmallVector<Value> optQubits;
    for (auto [refQubit, optQubit] : qubitMap) {
      optQubits.push_back(optQubit);
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, optQubits);
    return success();
  }
};

struct ConvertForOpOpt final : StatefulOpConversionPattern<scf::ForOp> {
  using StatefulOpConversionPattern<scf::ForOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    auto* region = op->getParentRegion();
    auto& qubitMap = getState().regionQubitMap[region];

    auto refQubits = getState().regionMap[op];
    SmallVector<Value> values;
    values.reserve(refQubits.size());
    for (auto qubit : refQubits) {
      values.push_back(qubit);
    }

    SmallVector<Value> optQubits;
    for (auto [refQubit, optQubit] : qubitMap) {
      optQubits.push_back(optQubit);
    }
    // Create a new for-loop with no iter_args
    auto newFor = rewriter.create<scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), ValueRange(optQubits));
    auto& srcBlock = op.getRegion().front();
    auto& dstBlock = newFor.getRegion().front();
    dstBlock.getOperations().splice(dstBlock.end(), srcBlock.getOperations());

    rewriter.setInsertionPointToEnd(&dstBlock);
    auto yield = dyn_cast<scf::YieldOp>(dstBlock.getTerminator());
    Operation* newYieldOp = nullptr;
    if (yield == nullptr) {
      newYieldOp = rewriter.create<scf::YieldOp>(op->getLoc(), values);
    } else {
      newYieldOp = rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, values);
    }
    newYieldOp->setAttr("needChange", rewriter.getStringAttr("yes"));
    auto& newRegion = newFor.getRegion();

    auto& regionQubitMap = getState().regionQubitMap[&newRegion];

    for (const auto& refQubit : refQubits) {
      regionQubitMap.try_emplace(refQubit, newRegion.getArgument(1));
    }

    rewriter.setInsertionPoint(op);

    auto& map = getState().regionQubitMap[op->getParentRegion()];
    for (size_t i = 0; i < newFor->getResults().size(); i++) {
      map[refQubits[i]] = newFor->getResult(i);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct MQTRefToMQTOpt final : impl::MQTRefToMQTOptBase<MQTRefToMQTOpt> {
  using MQTRefToMQTOptBase::MQTRefToMQTOptBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    LoweringState state;

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    MQTRefToMQTOptTypeConverter typeConverter(context);

    target.addIllegalDialect<ref::MQTRefDialect>();
    target.addLegalDialect<opt::MQTOptDialect>();

    target.addDynamicallyLegalOp<memref::AllocOp>(
        [&](memref::AllocOp op) { return !isQubitType(op); });
    target.addDynamicallyLegalOp<memref::DeallocOp>(
        [&](memref::DeallocOp op) { return !isQubitType(op); });
    target.addDynamicallyLegalOp<memref::LoadOp>(
        [&](memref::LoadOp op) { return !isQubitType(op); });
    target.addLegalOp<memref::StoreOp>();

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return !llvm::any_of(op->getOperandTypes(), [&](Type type) {
        return type == ref::QubitType::get(context);
      });
    });

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return !llvm::any_of(op.front().getArgumentTypes(), [&](Type type) {
        return type == ref::QubitType::get(context);
      });
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return !op->getAttrOfType<StringAttr>("needChange");
    });

    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });

    collectRegionQubits(module, &state, context);
    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    patterns.add<ConvertMQTRefMemRefAlloc>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefMemRefDealloc>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefMemRefLoad>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefAllocQubit>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefDeallocQubit>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefQubit>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefMeasure>(typeConverter, context, &state);
    patterns.add<ConvertMQTRefReset>(typeConverter, context, &state);
    patterns.add<ConvertCallOpOpt>(typeConverter, context, &state);
    patterns.add<ConvertFuncOpOpt>(typeConverter, context, &state);
    patterns.add<ConvertReturnOpOpt>(typeConverter, context, &state);
    patterns.add<ConvertIfOpOpt>(typeConverter, context, &state);
    patterns.add<ConvertYieldOpOpt>(typeConverter, context, &state);
    patterns.add<ConvertForOpOpt>(typeConverter, context, &state);
    ADD_CONVERT_PATTERN(GPhaseOp)
    ADD_CONVERT_PATTERN(IOp)
    ADD_CONVERT_PATTERN(BarrierOp)
    ADD_CONVERT_PATTERN(HOp)
    ADD_CONVERT_PATTERN(XOp)
    ADD_CONVERT_PATTERN(YOp)
    ADD_CONVERT_PATTERN(ZOp)
    ADD_CONVERT_PATTERN(SOp)
    ADD_CONVERT_PATTERN(SdgOp)
    ADD_CONVERT_PATTERN(TOp)
    ADD_CONVERT_PATTERN(TdgOp)
    ADD_CONVERT_PATTERN(VOp)
    ADD_CONVERT_PATTERN(VdgOp)
    ADD_CONVERT_PATTERN(UOp)
    ADD_CONVERT_PATTERN(U2Op)
    ADD_CONVERT_PATTERN(POp)
    ADD_CONVERT_PATTERN(SXOp)
    ADD_CONVERT_PATTERN(SXdgOp)
    ADD_CONVERT_PATTERN(ROp)
    ADD_CONVERT_PATTERN(RXOp)
    ADD_CONVERT_PATTERN(RYOp)
    ADD_CONVERT_PATTERN(RZOp)
    ADD_CONVERT_PATTERN(SWAPOp)
    ADD_CONVERT_PATTERN(iSWAPOp)
    ADD_CONVERT_PATTERN(iSWAPdgOp)
    ADD_CONVERT_PATTERN(PeresOp)
    ADD_CONVERT_PATTERN(PeresdgOp)
    ADD_CONVERT_PATTERN(DCXOp)
    ADD_CONVERT_PATTERN(ECROp)
    ADD_CONVERT_PATTERN(RXXOp)
    ADD_CONVERT_PATTERN(RYYOp)
    ADD_CONVERT_PATTERN(RZZOp)
    ADD_CONVERT_PATTERN(RZXOp)
    ADD_CONVERT_PATTERN(XXminusYYOp)
    ADD_CONVERT_PATTERN(XXplusYYOp)

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
    RewritePatternSet fixYieldPattern(context);
    ConversionTarget fixYieldTarget(*context);
    fixYieldTarget.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      return !(op->getAttrOfType<StringAttr>("moreChange"));
    });
    fixYieldPattern.add<ConvertYieldOpOpt2>(typeConverter, context, &state);

    if (failed(applyPartialConversion(module, fixYieldTarget,
                                      std::move(fixYieldPattern)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
