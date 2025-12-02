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
#include "llvm/ADT/SmallPtrSet.h"
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
  /// @brief Map each initial funcOp to its refQubits.
  llvm::DenseMap<Operation*, std::vector<Value>> regionMap;
  /// @brief Map each initial funcOp to its refQubits.
  llvm::DenseMap<Region*, llvm::DenseMap<Value, Value>> regionQubitMap;
  /// @brief Collect qubits of each region
  llvm::DenseMap<Region*, llvm::DenseSet<Value>> regionQubits;
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

llvm::DenseSet<Value> collectRegionQubits(Operation* op, LoweringState* state,
                                          MLIRContext* ctx) {

  auto regions = op->getRegions();
  DenseSet<Value> uniqueQubits;
  for (auto& region : regions) {
    auto& set = state->regionQubits[&region];

    if (region.empty()) {
      continue;
    }

    for (auto& operation : region.front().getOperations()) {
      if (operation.getNumRegions() > 0) {
        auto qubits = collectRegionQubits(&operation, state, ctx);
        for (auto qubit : qubits) {
          uniqueQubits.insert(qubit);
          set.insert(qubit);
        }
      }
      for (auto operand : operation.getOperands()) {
        if (operand.getType() == ref::QubitType::get(ctx)) {
          uniqueQubits.insert(operand);
          set.insert(operand);
        }
      }
      for (auto result : operation.getResults()) {
        if (result.getType() == ref::QubitType::get(ctx)) {
          uniqueQubits.insert(result);
          set.insert(result);
        }
      }
    }
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
    auto regionOp = op->getParentRegion();
    auto maps = this->getState().regionQubitMap[regionOp];
    auto funcOp = op->template getParentOfType<mlir::func::FuncOp>();
    funcOp.print(llvm::outs());
    for (const auto& [key, value] : maps) {
      llvm::outs() << key << " = " << value << "\n";
    }
    auto mapQubits2 = [&](const auto& refQubits) {
      std::vector<Value> optQubits;
      for (const auto& refQubit : refQubits) {
        optQubits.emplace_back(
            this->getState().regionQubitMap[regionOp][refQubit]);
      }
      return optQubits;
    };

    auto optInQubits = mapQubits2(op.getInQubits());
    auto optPosCtrlQubitsValues = mapQubits2(op.getPosCtrlInQubits());
    auto optNegCtrlQubitsValues = mapQubits2(op.getNegCtrlInQubits());

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
      //  this->getState().qubitMap[op.getAllInQubits()[i]] = optResults[i];
      this->getState().regionQubitMap[regionOp][op.getAllInQubits()[i]] =
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
    /*

      auto newIf = rewriter.create<scf::IfOp>(op->getLoc(), TypeRange{optType},
                                              op.getCondition(), true);
      newIf->setAttr("needChange", rewriter.getStringAttr("no"));

      //

      Value operand;
      auto& ops = op.getThenRegion().front().getOperations();
      for (auto& i : ops) {
        auto hOp = dyn_cast<ref::HOp>(i);
        if (hOp) {
          operand = hOp->getOperand(0);
          auto optQubit =
              getState().regionQubitMap[op->getParentRegion()][operand];
          getState().regionQubitMap[&newIf.getThenRegion()].try_emplace(operand,
                                                                        optQubit);
        }
      }

      rewriter.eraseBlock(&newIf.getThenRegion().front());
      rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                  newIf.getThenRegion().end());
      if (!op.getElseRegion().empty()) {
        rewriter.eraseBlock(&newIf.getElseRegion().front());
        rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(),
                                    newIf.getElseRegion().end());
      } else {
        rewriter.setInsertionPointToEnd(&newIf.getElseRegion().back());
        rewriter.create<scf::YieldOp>(op.getLoc(), operand);
        //  auto* elseTerminator = newIf.getElseRegion().back().getTerminator();
        // rewriter.replaceOpWithNewOp<scf::YieldOp>(elseTerminator, operand);
      }
      auto* thenTerminator = newIf.getThenRegion().back().getTerminator();
      rewriter.setInsertionPointToEnd(&newIf.getThenRegion().back());
      rewriter.eraseOp(thenTerminator);
      rewriter.create<scf::YieldOp>(op.getLoc(), operand);
  */
    /*
      auto alloc = rewriter.create<opt::AllocQubitOp>(op->getLoc());
      auto const optType = opt::QubitType::get(rewriter.getContext());
      auto newIf = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange{optType},
                                              adaptor.getCondition(), false);
      rewriter.setInsertionPointToEnd(&newIf.getThenRegion().back());
      rewriter.create<scf::YieldOp>(op.getLoc(), alloc->getResult(0));
      /*
      rewriter.setInsertionPointToEnd(&newIf.getElseRegion().back());
      rewriter.create<scf::YieldOp>(op.getLoc(), alloc->getResult(0));
      */

    auto b = rewriter.create<ref::AllocQubitOp>(op->getLoc());

    auto newIf = rewriter.create<scf::IfOp>(op->getLoc(), ValueRange{},
                                            adaptor.getCondition(), false);
    newIf->setAttr("needChange", rewriter.getStringAttr("no"));

    rewriter.replaceOp(op, newIf->getResults());
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    funcOp.print(llvm::outs());
    return success();
    // inline the regions
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
    /*
        target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
          if (op->getAttrOfType<StringAttr>("needChange")) {

            return true;
          }
          for (auto& opt : op.getThenRegion().front()) {
            for (auto operand : opt.getOperands()) {
              if (operand.getType() == ref::QubitType::get(context)) {
                return false;
              }
            }
          }
          return true;
        });
        */
    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      return op->getAttrOfType<StringAttr>("needChange") != nullptr;
    });
    /*
    target.addDynamicallyLegalOp<scf::IfOp>(
        [&](scf::IfOp op) { return op->getResults().size() > 0; });
    }}*/
    collectRegionQubits(module, &state, context);
    auto map = state.regionQubits;
    for (auto [region, qubits] : map) {

      auto* op = region->getParentOp();
      llvm::outs() << "current op ";
      region->front().print(llvm::outs());
      llvm::outs() << "\n";
      for (auto qubit : qubits) {
        qubit.print(llvm::outs());
        llvm::outs() << "value\n";
      }
    }

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
    /*
        if (failed(applyPartialConversion(module, target, std::move(patterns))))
       { signalPassFailure();
        }
        */
  };
};

} // namespace mqt::ir
