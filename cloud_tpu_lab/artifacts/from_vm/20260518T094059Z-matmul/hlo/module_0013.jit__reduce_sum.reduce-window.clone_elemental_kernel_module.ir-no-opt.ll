; ModuleID = '__compute_module_reduce-window.clone_elemental_kernel_module'
source_filename = "__compute_module_reduce-window.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@0 = private unnamed_addr constant [4 x i8] zeroinitializer, align 4
@reduce-window.clone_parallel_bounds = private constant [4 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 4]], [1 x [2 x i64]] [[2 x i64] [i64 4, i64 8]], [1 x [2 x i64]] [[2 x i64] [i64 8, i64 12]], [1 x [2 x i64]] [[2 x i64] [i64 12, i64 16]]]

; Function Attrs: uwtable
define ptr @reduce-window.clone_kernel(ptr %0) #0 {
  %reducer_function_parameter_addresses = alloca ptr, i32 2, align 8
  %reducer_function_return_value_addr = alloca float, align 4
  %arg_addr8 = alloca float, align 4
  %arg_addr = alloca float, align 4
  %reduce-window.clone.invar_address.window.1 = alloca i64, align 8
  %reduce-window.clone.invar_address.window.0 = alloca i64, align 8
  %reduce_window_accum_ptr = alloca float, align 4
  %reduce-window.clone.invar_address.dim.1 = alloca i64, align 8
  %reduce-window.clone.invar_address.dim.0 = alloca i64, align 8
  %num_workgroups_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %num_workgroups = load ptr, ptr %num_workgroups_gep, align 8
  %num_workgroups_x_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 0
  %num_workgroups_y_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 1
  %num_workgroups_z_gep = getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, ptr %num_workgroups, i32 0, i32 2
  %num_workgroups_x = load i64, ptr %num_workgroups_x_gep, align 4
  %num_workgroups_y = load i64, ptr %num_workgroups_y_gep, align 4
  %num_workgroups_z = load i64, ptr %num_workgroups_z_gep, align 4
  %workgroup_id_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 0
  %workgroup_id_y_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 1
  %workgroup_id_z_gep = getelementptr inbounds nuw %XLA_CPU_WorkGroupId, ptr %workgroup_id, i32 0, i32 2
  %workgroup_id_x = load i64, ptr %workgroup_id_x_gep, align 4
  %workgroup_id_y = load i64, ptr %workgroup_id_y_gep, align 4
  %workgroup_id_z = load i64, ptr %workgroup_id_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %lo_dim_0_gep = getelementptr inbounds [4 x [1 x [2 x i64]]], ptr @reduce-window.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [4 x [1 x [2 x i64]]], ptr @reduce-window.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  store i64 %lo_dim_0, ptr %reduce-window.clone.invar_address.dim.0, align 4
  br label %reduce-window.clone.loop_header.dim.0

reduce-window.clone.loop_header.dim.0:            ; preds = %reduce-window.clone.loop_exit.dim.1, %1
  %reduce-window.clone.indvar.dim.0 = load i64, ptr %reduce-window.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %reduce-window.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %reduce-window.clone.loop_exit.dim.0, label %reduce-window.clone.loop_body.dim.0

reduce-window.clone.loop_body.dim.0:              ; preds = %reduce-window.clone.loop_header.dim.0
  store i64 0, ptr %reduce-window.clone.invar_address.dim.1, align 4
  br label %reduce-window.clone.loop_header.dim.1

reduce-window.clone.loop_header.dim.1:            ; preds = %reduce-window.clone.loop_exit.window.0, %reduce-window.clone.loop_body.dim.0
  %reduce-window.clone.indvar.dim.1 = load i64, ptr %reduce-window.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %reduce-window.clone.indvar.dim.1, 16
  br i1 %3, label %reduce-window.clone.loop_exit.dim.1, label %reduce-window.clone.loop_body.dim.1

reduce-window.clone.loop_body.dim.1:              ; preds = %reduce-window.clone.loop_header.dim.1
  %4 = load float, ptr %arg1, align 4, !invariant.load !1, !noalias !6
  store float %4, ptr %reduce_window_accum_ptr, align 4
  store i64 0, ptr %reduce-window.clone.invar_address.window.0, align 4
  br label %reduce-window.clone.loop_header.window.0

reduce-window.clone.loop_header.window.0:         ; preds = %reduce-window.clone.loop_exit.window.1, %reduce-window.clone.loop_body.dim.1
  %reduce-window.clone.indvar.window.0 = load i64, ptr %reduce-window.clone.invar_address.window.0, align 4
  %5 = icmp uge i64 %reduce-window.clone.indvar.window.0, 32
  br i1 %5, label %reduce-window.clone.loop_exit.window.0, label %reduce-window.clone.loop_body.window.0

reduce-window.clone.loop_body.window.0:           ; preds = %reduce-window.clone.loop_header.window.0
  store i64 0, ptr %reduce-window.clone.invar_address.window.1, align 4
  br label %reduce-window.clone.loop_header.window.1

reduce-window.clone.loop_header.window.1:         ; preds = %in_bounds-after, %reduce-window.clone.loop_body.window.0
  %reduce-window.clone.indvar.window.1 = load i64, ptr %reduce-window.clone.invar_address.window.1, align 4
  %6 = icmp uge i64 %reduce-window.clone.indvar.window.1, 32
  br i1 %6, label %reduce-window.clone.loop_exit.window.1, label %reduce-window.clone.loop_body.window.1

reduce-window.clone.loop_body.window.1:           ; preds = %reduce-window.clone.loop_header.window.1
  %7 = mul nsw i64 %reduce-window.clone.indvar.dim.0, 32
  %8 = mul nsw i64 %reduce-window.clone.indvar.window.0, 1
  %9 = add nsw i64 %7, %8
  %10 = sub nsw i64 %9, 0
  %11 = srem i64 %10, 1
  %12 = icmp eq i64 %11, 0
  %13 = and i1 true, %12
  %14 = sdiv i64 %10, 1
  %15 = icmp ult i64 %14, 512
  %16 = and i1 %13, %15
  %17 = mul nsw i64 %reduce-window.clone.indvar.dim.1, 32
  %18 = mul nsw i64 %reduce-window.clone.indvar.window.1, 1
  %19 = add nsw i64 %17, %18
  %20 = sub nsw i64 %19, 0
  %21 = srem i64 %20, 1
  %22 = icmp eq i64 %21, 0
  %23 = and i1 %16, %22
  %24 = sdiv i64 %20, 1
  %25 = icmp ult i64 %24, 512
  %26 = and i1 %23, %25
  br i1 %26, label %in_bounds-true, label %in_bounds-false

in_bounds-after:                                  ; preds = %in_bounds-false, %in_bounds-true
  %invar.inc7 = add nuw nsw i64 %reduce-window.clone.indvar.window.1, 1
  store i64 %invar.inc7, ptr %reduce-window.clone.invar_address.window.1, align 4
  br label %reduce-window.clone.loop_header.window.1

reduce-window.clone.loop_exit.window.1:           ; preds = %reduce-window.clone.loop_header.window.1
  %invar.inc6 = add nuw nsw i64 %reduce-window.clone.indvar.window.0, 1
  store i64 %invar.inc6, ptr %reduce-window.clone.invar_address.window.0, align 4
  br label %reduce-window.clone.loop_header.window.0

reduce-window.clone.loop_exit.window.0:           ; preds = %reduce-window.clone.loop_header.window.0
  %27 = load float, ptr %reduce_window_accum_ptr, align 4
  %28 = getelementptr inbounds [16 x [16 x float]], ptr %arg2, i64 0, i64 %reduce-window.clone.indvar.dim.0, i64 %reduce-window.clone.indvar.dim.1
  store float %27, ptr %28, align 4, !alias.scope !6
  %invar.inc5 = add nuw nsw i64 %reduce-window.clone.indvar.dim.1, 1
  store i64 %invar.inc5, ptr %reduce-window.clone.invar_address.dim.1, align 4
  br label %reduce-window.clone.loop_header.dim.1

reduce-window.clone.loop_exit.dim.1:              ; preds = %reduce-window.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %reduce-window.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %reduce-window.clone.invar_address.dim.0, align 4
  br label %reduce-window.clone.loop_header.dim.0, !llvm.loop !9

reduce-window.clone.loop_exit.dim.0:              ; preds = %reduce-window.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %reduce-window.clone.loop_exit.dim.0
  ret ptr null

in_bounds-true:                                   ; preds = %reduce-window.clone.loop_body.window.1
  %29 = getelementptr inbounds [512 x [512 x float]], ptr %arg0, i64 0, i64 %14, i64 %24
  %30 = load float, ptr %29, align 4, !invariant.load !1, !noalias !6
  %31 = load float, ptr %reduce_window_accum_ptr, align 4
  store float %31, ptr %arg_addr, align 4
  store float %30, ptr %arg_addr8, align 4
  %32 = getelementptr inbounds ptr, ptr %reducer_function_parameter_addresses, i64 0
  store ptr %arg_addr, ptr %32, align 8
  %33 = getelementptr inbounds ptr, ptr %reducer_function_parameter_addresses, i64 1
  store ptr %arg_addr8, ptr %33, align 8
  call void @region_0.7(ptr %reducer_function_return_value_addr, ptr null, ptr %reducer_function_parameter_addresses, ptr null, ptr null, ptr null)
  %34 = load float, ptr %reducer_function_return_value_addr, align 4
  store float %34, ptr %reduce_window_accum_ptr, align 4
  br label %in_bounds-after

in_bounds-false:                                  ; preds = %reduce-window.clone.loop_body.window.1
  br label %in_bounds-after
}

; Function Attrs: alwaysinline uwtable
define internal void @region_0.7(ptr %retval, ptr noalias %run_options, ptr noalias %params, ptr noalias %buffer_table, ptr noalias %status, ptr noalias %prof_counters) #1 {
entry:
  %add.6 = alloca float, align 4
  %0 = getelementptr inbounds ptr, ptr %params, i64 0
  %Arg_0.4 = load ptr, ptr %0, align 8, !dereferenceable !4, !align !4
  %1 = getelementptr inbounds ptr, ptr %params, i64 1
  %Arg_1.5 = load ptr, ptr %1, align 8, !dereferenceable !4, !align !4
  %2 = load float, ptr %Arg_0.4, align 4, !alias.scope !11, !noalias !14
  %3 = load float, ptr %Arg_1.5, align 4, !alias.scope !16, !noalias !14
  %add.61 = fadd reassoc float %2, %3
  store float %add.61, ptr %add.6, align 4, !alias.scope !14
  %load_ret_value = load float, ptr %add.6, align 4
  store float %load_ret_value, ptr %retval, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline uwtable "denormal-fp-math"="preserve-sign" "no-frame-pointer-elim"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 1}
!1 = !{}
!2 = !{i64 1048576}
!3 = !{i64 64}
!4 = !{i64 4}
!5 = !{i64 1024}
!6 = !{!7}
!7 = !{!"result slice: {index:9, offset:1048576, size:1024}", !8}
!8 = !{!"XLA host kernel reduce-window.clone_kernel AA domain"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = !{!12}
!12 = !{!"buffer: {index:4, offset:0, size:4}", !13}
!13 = !{!"XLA global AA domain"}
!14 = !{!15}
!15 = !{!"buffer: {index:6, offset:0, size:4}", !13}
!16 = !{!17}
!17 = !{!"buffer: {index:5, offset:0, size:4}", !13}
