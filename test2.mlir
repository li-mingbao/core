// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() {
    %0 = mqtref.allocQubit
    call @test(%0) : (!mqtref.Qubit) -> ()
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    scf.while : () -> () {
      mqtref.h() %0
      scf.while : () -> () {
        mqtref.h() %0
        scf.condition(%true)
      } do {
        mqtref.h() %0
        scf.yield
      }
      scf.condition(%true)
    } do {
      mqtref.h() %0
      scf.yield
    }
    mqtref.h() %0
    scf.for %arg0 = %c1 to %c1 step %c1 {
      mqtref.h() %0
    }
    %1 = mqtref.allocQubit
    scf.if %true {
      mqtref.h() %0
    } else {
      mqtref.x() %1
    }
    mqtref.h() %0
    mqtref.h() %1
    return
  }
  func.func @test(%arg0: !mqtref.Qubit) {
    return
  }
}
