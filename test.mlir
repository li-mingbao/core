// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() {

    %q0 = "mqtopt.allocQubit"() : () -> !mqtopt.Qubit
    %q1 = func.call @test(%q0) : (!mqtopt.Qubit) -> !mqtopt.Qubit
    %c1 = arith.constant 1 : index
    %condition = arith.constant 1 : i1
    %q2 = scf.while(%arg1 = %q1) : (!mqtopt.Qubit) -> !mqtopt.Qubit {

      %b = mqtopt.h() %arg1 : !mqtopt.Qubit
      %3 = scf.while(%arg3 = %b) : (!mqtopt.Qubit) -> !mqtopt.Qubit {
        %c = mqtopt.h() %arg3 : !mqtopt.Qubit
        scf.condition(%condition) %c : !mqtopt.Qubit
     } do {
      ^bb0(%arg4: !mqtopt.Qubit):
        %j = mqtopt.h() %arg4 : !mqtopt.Qubit
        scf.yield %j : !mqtopt.Qubit
      }
      scf.condition(%condition) %b : !mqtopt.Qubit
    } do {
     ^bb0(%arg2: !mqtopt.Qubit):
     %b = mqtopt.h() %arg2 : !mqtopt.Qubit
     scf.yield %b : !mqtopt.Qubit
    }
    %q3 = mqtopt.h() %q2 : !mqtopt.Qubit
    %q4 = scf.for %iv = %c1 to %c1 step %c1
      iter_args(%arg1 = %q3) -> (!mqtopt.Qubit) {
        %b = mqtopt.h() %arg1 : !mqtopt.Qubit
        scf.yield %b : !mqtopt.Qubit
    }
    %q5 = scf.if %condition -> (!mqtopt.Qubit){
        %b = mqtopt.h() %q4 : !mqtopt.Qubit
        scf.yield %b : !mqtopt.Qubit
    } else {
        %c = mqtopt.x() %q4 : !mqtopt.Qubit
        scf.yield %c : !mqtopt.Qubit

    }
    %q6 = mqtopt.h() %q5 : !mqtopt.Qubit
    return
  }

  func.func @test(%q : !mqtopt.Qubit) -> !mqtopt.Qubit {
    func.return %q : !mqtopt.Qubit
  }
}
