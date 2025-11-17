/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any opt gate operation to the same
// gate operation in the ref dialect
#include "mlir/Transforms/RegionUtils.h"
#define ADD_CONVERT_PATTERN(gate)                                              \
  patterns                                                                     \
      .add<ConvertMQTOptGateOp<::mqt::ir::opt::gate, ::mqt::ir::ref::gate>>(   \
          typeConverter, context);

#include "mlir/Conversion/MQTOptToMQTRef/MQTOptToMQTRef.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTRef/IR/MQTRefDialect.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_MQTOPTTOMQTREF
#include "mlir/Conversion/MQTOptToMQTRef/MQTOptToMQTRef.h.inc"

namespace {

bool isQubitType(const MemRefType type) {
  return llvm::isa<opt::QubitType>(type.getElementType());
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

bool isQubitType(memref::StoreOp op) {
  const auto& memRef = op.getMemref();
  const auto& memRefType = llvm::cast<MemRefType>(memRef.getType());
  return isQubitType(memRefType);
}

} // namespace

class MQTOptToMQTRefTypeConverter final : public TypeConverter {
public:
  explicit MQTOptToMQTRefTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });

    // QubitType conversion
    addConversion([ctx](opt::QubitType /*type*/) -> Type {
      return ref::QubitType::get(ctx);
    });

    // MemRefType conversion
    addConversion([ctx](MemRefType type) -> Type {
      if (isQubitType(type)) {
        return MemRefType::get(type.getShape(), ref::QubitType::get(ctx));
      }
      return type;
    });
  }
};

struct ConvertMQTOptMemRefAlloc final : OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    const auto& qubitType = ref::QubitType::get(rewriter.getContext());
    const auto& memRefType =
        MemRefType::get(op.getType().getShape(), qubitType);

    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memRefType,
                                                 adaptor.getDynamicSizes());

    return success();
  }
};

struct ConvertMQTOptMemRefDealloc final
    : OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, adaptor.getMemref());
    return success();
  }
};

struct ConvertMQTOptAllocQubit final : OpConversionPattern<opt::AllocQubitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(const opt::AllocQubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ref::AllocQubitOp>(op);
    return success();
  }
};

struct ConvertMQTOptDeallocQubit final
    : OpConversionPattern<opt::DeallocQubitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(const opt::DeallocQubitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<ref::DeallocQubitOp>(op, adaptor.getQubit());
    return success();
  }
};

struct ConvertMQTOptMemRefLoad final : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getMemref(),
                                                adaptor.getIndices());

    return success();
  }
};

struct ConvertMQTOptMemRefStore final : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    if (!isQubitType(op)) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMQTOptMeasure final : OpConversionPattern<opt::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& refQubit = adaptor.getInQubit();

    // create new operation
    auto measure = rewriter.create<ref::MeasureOp>(
        op.getLoc(), op.getOutBit().getType(), refQubit);

    // replace the results of the old operation with the new results and
    // delete old operation
    rewriter.replaceOp(op, {refQubit, measure.getOutBit()});
    return success();
  }
};

struct ConvertMQTOptReset final : OpConversionPattern<opt::ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& refQubit = adaptor.getInQubit();

    // create new operation
    rewriter.create<ref::ResetOp>(op.getLoc(), refQubit);

    // replace the results of the old operation with the new results and
    // delete old operation
    rewriter.replaceOp(op, refQubit);
    return success();
  }
};

struct ConvertMQTOptQubit final : OpConversionPattern<opt::QubitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(opt::QubitOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto& qubitType = ref::QubitType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<ref::QubitOp>(op, qubitType, op.getIndex());
    return success();
  }
};

template <typename MQTGateOptOp, typename MQTGateRefOp>
struct ConvertMQTOptGateOp final : OpConversionPattern<MQTGateOptOp> {
  using OpConversionPattern<MQTGateOptOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateOptOp op, typename MQTGateOptOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // get all the input qubits including the ctrl qubits
    const auto& refInQubitsValues = adaptor.getInQubits();
    const auto& refPosCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto& refNegCtrlQubitsValues = adaptor.getNegCtrlInQubits();

    // append them to a single vector
    SmallVector<Value> refQubitsValues;
    refQubitsValues.reserve(refInQubitsValues.size() +
                            refPosCtrlQubitsValues.size() +
                            refNegCtrlQubitsValues.size());
    refQubitsValues.append(refInQubitsValues.begin(), refInQubitsValues.end());
    refQubitsValues.append(refPosCtrlQubitsValues.begin(),
                           refPosCtrlQubitsValues.end());
    refQubitsValues.append(refNegCtrlQubitsValues.begin(),
                           refNegCtrlQubitsValues.end());

    // get the static params and paramMask if they exist
    auto staticParams = op.getStaticParams()
                            ? DenseF64ArrayAttr::get(rewriter.getContext(),
                                                     *op.getStaticParams())
                            : DenseF64ArrayAttr{};
    auto paramMask = op.getParamsMask()
                         ? DenseBoolArrayAttr::get(rewriter.getContext(),
                                                   *op.getParamsMask())
                         : DenseBoolArrayAttr{};

    // create new operation
    rewriter.create<MQTGateRefOp>(
        op.getLoc(), staticParams, paramMask, op.getParams(), refInQubitsValues,
        refPosCtrlQubitsValues, refNegCtrlQubitsValues);

    // replace the results of the old operation with the new results and
    // delete old operation
    rewriter.replaceOp(op, refQubitsValues);

    return success();
  }
};

struct ConvertCallOp final : OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;

    for (auto type : op->getResultTypes()) {
      if (!isa<opt::QubitType>(type)) {
        resultTypes.push_back(type);
      }
    }
    rewriter.create<func::CallOp>(op->getLoc(), adaptor.getCallee(),
                                  resultTypes, adaptor.getOperands());

    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

struct ConvertFuncOp final : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    SmallVector<Type> argumentTypes;
    auto refType = ref::QubitType::get(rewriter.getContext());

    for (auto blockArg : op.front().getArguments()) {
      if (isa<opt::QubitType>(blockArg.getType())) {
        blockArg.setType(refType);
        argumentTypes.push_back(refType);
      } else {
        argumentTypes.push_back(blockArg.getType());
      }
    }

    SmallVector<Type> resultTypes;
    for (auto type : op->getResultTypes()) {
      if (!isa<opt::QubitType>(type)) {
        resultTypes.push_back(type);
      }
    }
    auto newFuncType =
        rewriter.getFunctionType(argumentTypes, resultTypes); // void
    op.setFunctionType(newFuncType);
    return success();
  }
};

struct ConvertReturnOp final : OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    rewriter.create<func::ReturnOp>(op->getLoc());
    rewriter.eraseOp(op);

    return success();
  }
};
struct ConvertForOp final : OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // Create a new for-loop with no iter_args
    auto newFor = rewriter.create<scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), ValueRange{});

    // Map induction variable only (no iter_args)
    IRMapping mapping;
    mapping.map(op.getInductionVar(), newFor.getInductionVar());

    Block* newBlock = newFor.getBody();
    for (auto operation : op.getRegionIterArgs()) {
      if (operation.getType() == opt::QubitType::get(rewriter.getContext())) {
        operation.replaceAllUsesWith(adaptor.getInitArgs()[0]);
      }
    }

    rewriter.setInsertionPoint(newBlock->getTerminator());
    // Clone body operations except for scf.yield
    for (Operation& oldOp : op.getBody()->getOperations()) {
      if (!llvm::isa<scf::YieldOp>(oldOp)) {
        rewriter.clone(oldOp, mapping);
      }
    }
    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    funcOp.print(llvm::outs());
    rewriter.replaceOp(op, adaptor.getInitArgs());

    return success();
  }
};
struct ConvertYieldOp final : OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    // rewriter.eraseOp(op);

    return success();
  }
};
struct MQTOptToMQTRef final : impl::MQTOptToMQTRefBase<MQTOptToMQTRef> {
  using MQTOptToMQTRefBase::MQTOptToMQTRefBase;

  void convertLoopRecursively(Operation* op, ConversionTarget& target,
                              FrozenRewritePatternSet& patterns) {
    for (Region& region : op->getRegions()) {
      for (Block& block : region) {         // bottom-up block order
        for (Operation& nestedOp : block) { // bottom-up op order
          convertLoopRecursively(&nestedOp, target, patterns);
        }
      }
    }

    ConversionConfig config;
    config.allowPatternRollback = false;
    (void)applyPartialConversion(op, target, patterns, config);
  }
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    MQTOptToMQTRefTypeConverter typeConverter(context);

    target.addIllegalDialect<opt::MQTOptDialect>();
    target.addLegalDialect<ref::MQTRefDialect>();
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return !llvm::any_of(op->getOperandTypes(), [&](Type type) {
        return type == opt::QubitType::get(context);
      });
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return !llvm::any_of(op.front().getArgumentTypes(), [&](Type type) {
        return type == opt::QubitType::get(context);
      });
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return !llvm::any_of(op->getOperandTypes(), [&](Type type) {
        return type == opt::QubitType::get(context) ||
               type == ref::QubitType::get(context);
      });
    });
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      return !llvm::any_of(op.getRegionIterArgs(), [&](BlockArgument type) {
        return type.getType() == opt::QubitType::get(context);
      });
    });
    target.addDynamicallyLegalOp<memref::AllocOp>(
        [&](memref::AllocOp op) { return !isQubitType(op); });
    target.addDynamicallyLegalOp<memref::DeallocOp>(
        [&](memref::DeallocOp op) { return !isQubitType(op); });
    target.addDynamicallyLegalOp<memref::LoadOp>(
        [&](memref::LoadOp op) { return !isQubitType(op); });
    target.addDynamicallyLegalOp<memref::StoreOp>(
        [&](memref::StoreOp op) { return !isQubitType(op); });
    target.addIllegalOp<scf::YieldOp>();
    patterns.add<ConvertMQTOptMemRefAlloc, ConvertMQTOptMemRefDealloc,
                 ConvertMQTOptMemRefStore, ConvertMQTOptMemRefLoad,
                 ConvertMQTOptAllocQubit, ConvertMQTOptDeallocQubit,
                 ConvertMQTOptQubit, ConvertMQTOptMeasure, ConvertMQTOptReset>(
        typeConverter, context);
    patterns.add<ConvertCallOp, ConvertReturnOp, ConvertFuncOp, ConvertForOp,
                 ConvertYieldOp>(typeConverter, context);
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
    ConversionConfig config;
    config.allowPatternRollback = false;
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    convertLoopRecursively(module, target, frozenPatterns);
  };
};

} // namespace mqt::ir
