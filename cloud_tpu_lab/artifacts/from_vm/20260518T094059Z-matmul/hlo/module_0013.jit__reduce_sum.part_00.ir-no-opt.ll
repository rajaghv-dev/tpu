; ModuleID = '__compute_module_part_00'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.1 = private unnamed_addr constant [4 x i8] zeroinitializer

; Function Attrs: uwtable
define ptr @reduce_convert_fusion(ptr %0) #0 {
  %reduce_function_parameter_addresses = alloca ptr, i32 2, align 8
  %reduce_function_return_value_addr = alloca float, align 4
  %arg_addr4 = alloca float, align 4
  %arg_addr = alloca float, align 4
  %reduce.1.inner.invar_address.reduction_dim.1 = alloca i64, align 8
  %reduce.1.inner.invar_address.reduction_dim.0 = alloca i64, align 8
  %accumulator_0 = alloca float, align 4
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
  %constant.0 = load float, ptr @__llvmsplit_unnamed.1, align 4
  store float %constant.0, ptr %accumulator_0, align 4
  store i64 0, ptr %reduce.1.inner.invar_address.reduction_dim.0, align 4
  br label %reduce.1.inner.loop_header.reduction_dim.0

reduce.1.inner.loop_header.reduction_dim.0:       ; preds = %reduce.1.inner.loop_exit.reduction_dim.1, %1
  %reduce.1.inner.indvar.reduction_dim.0 = load i64, ptr %reduce.1.inner.invar_address.reduction_dim.0, align 4
  %2 = icmp uge i64 %reduce.1.inner.indvar.reduction_dim.0, 16
  br i1 %2, label %reduce.1.inner.loop_exit.reduction_dim.0, label %reduce.1.inner.loop_body.reduction_dim.0

reduce.1.inner.loop_body.reduction_dim.0:         ; preds = %reduce.1.inner.loop_header.reduction_dim.0
  store i64 0, ptr %reduce.1.inner.invar_address.reduction_dim.1, align 4
  br label %reduce.1.inner.loop_header.reduction_dim.1

reduce.1.inner.loop_header.reduction_dim.1:       ; preds = %reduce.1.inner.loop_body.reduction_dim.1, %reduce.1.inner.loop_body.reduction_dim.0
  %reduce.1.inner.indvar.reduction_dim.1 = load i64, ptr %reduce.1.inner.invar_address.reduction_dim.1, align 4
  %3 = icmp uge i64 %reduce.1.inner.indvar.reduction_dim.1, 16
  br i1 %3, label %reduce.1.inner.loop_exit.reduction_dim.1, label %reduce.1.inner.loop_body.reduction_dim.1

reduce.1.inner.loop_body.reduction_dim.1:         ; preds = %reduce.1.inner.loop_header.reduction_dim.1
  %4 = load float, ptr %accumulator_0, align 4
  %5 = getelementptr inbounds [16 x [16 x float]], ptr %arg0, i64 0, i64 %reduce.1.inner.indvar.reduction_dim.0, i64 %reduce.1.inner.indvar.reduction_dim.1
  %6 = load float, ptr %5, align 4, !invariant.load !1, !noalias !5
  store float %4, ptr %arg_addr, align 4
  store float %6, ptr %arg_addr4, align 4
  %7 = getelementptr inbounds ptr, ptr %reduce_function_parameter_addresses, i64 0
  store ptr %arg_addr, ptr %7, align 8
  %8 = getelementptr inbounds ptr, ptr %reduce_function_parameter_addresses, i64 1
  store ptr %arg_addr4, ptr %8, align 8
  call void @reduce.1(ptr %reduce_function_return_value_addr, ptr null, ptr %reduce_function_parameter_addresses, ptr null, ptr null, ptr null)
  %9 = load float, ptr %reduce_function_return_value_addr, align 4
  store float %9, ptr %accumulator_0, align 4
  %invar.inc3 = add nuw nsw i64 %reduce.1.inner.indvar.reduction_dim.1, 1
  store i64 %invar.inc3, ptr %reduce.1.inner.invar_address.reduction_dim.1, align 4
  br label %reduce.1.inner.loop_header.reduction_dim.1

reduce.1.inner.loop_exit.reduction_dim.1:         ; preds = %reduce.1.inner.loop_header.reduction_dim.1
  %invar.inc = add nuw nsw i64 %reduce.1.inner.indvar.reduction_dim.0, 1
  store i64 %invar.inc, ptr %reduce.1.inner.invar_address.reduction_dim.0, align 4
  br label %reduce.1.inner.loop_header.reduction_dim.0

reduce.1.inner.loop_exit.reduction_dim.0:         ; preds = %reduce.1.inner.loop_header.reduction_dim.0
  %10 = load float, ptr %accumulator_0, align 4
  %11 = bitcast float %10 to i32
  %12 = lshr i32 %11, 16
  %13 = and i32 %12, 1
  %14 = add i32 32767, %13
  %15 = call i1 @llvm.is.fpclass.f32(float %10, i32 3)
  %16 = and i32 %11, -4194304
  %17 = or i32 %16, 4194304
  %18 = add i32 %11, %14
  %19 = select i1 %15, i32 %17, i32 %18
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i16
  %22 = bitcast i16 %21 to bfloat
  store bfloat %22, ptr %arg1, align 2, !alias.scope !5
  br label %return

return:                                           ; preds = %reduce.1.inner.loop_exit.reduction_dim.0
  ret ptr null
}

; Function Attrs: alwaysinline uwtable
define internal void @reduce.1(ptr %retval, ptr noalias %run_options, ptr noalias %params, ptr noalias %buffer_table, ptr noalias %status, ptr noalias %prof_counters) #1 {
entry:
  %add.0 = alloca float, align 4
  %0 = getelementptr inbounds ptr, ptr %params, i64 0
  %Arg_0.0 = load ptr, ptr %0, align 8, !dereferenceable !8, !align !8
  %1 = getelementptr inbounds ptr, ptr %params, i64 1
  %Arg_1.0 = load ptr, ptr %1, align 8, !dereferenceable !8, !align !8
  %2 = load float, ptr %Arg_0.0, align 4, !alias.scope !9, !noalias !12
  %3 = load float, ptr %Arg_1.0, align 4, !alias.scope !14, !noalias !12
  %add.01 = fadd reassoc float %2, %3
  store float %add.01, ptr %add.0, align 4, !alias.scope !12
  %load_ret_value = load float, ptr %add.0, align 4
  store float %load_ret_value, ptr %retval, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f32(float, i32 immarg) #2

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { alwaysinline uwtable "denormal-fp-math"="preserve-sign" "frame-pointer"="none" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 1024}
!3 = !{i64 64}
!4 = !{i64 2}
!5 = !{!6}
!6 = !{!"result slice: {index:2, offset:0, size:2}", !7}
!7 = !{!"XLA host kernel reduce_convert_fusion AA domain"}
!8 = !{i64 4}
!9 = !{!10}
!10 = !{!"buffer: {index:7, offset:0, size:4}", !11}
!11 = !{!"XLA global AA domain"}
!12 = !{!13}
!13 = !{!"buffer: {index:3, offset:0, size:4}", !11}
!14 = !{!15}
!15 = !{!"buffer: {index:8, offset:0, size:4}", !11}
