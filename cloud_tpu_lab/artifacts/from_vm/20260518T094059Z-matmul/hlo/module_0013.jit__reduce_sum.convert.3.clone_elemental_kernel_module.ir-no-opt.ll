; ModuleID = '__compute_module_convert.3.clone_elemental_kernel_module'
source_filename = "__compute_module_convert.3.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@0 = private unnamed_addr constant [4 x i8] zeroinitializer, align 4
@convert.3.clone_parallel_bounds = private constant [5 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 102]], [1 x [2 x i64]] [[2 x i64] [i64 102, i64 204]], [1 x [2 x i64]] [[2 x i64] [i64 204, i64 306]], [1 x [2 x i64]] [[2 x i64] [i64 306, i64 408]], [1 x [2 x i64]] [[2 x i64] [i64 408, i64 512]]]

; Function Attrs: uwtable
define ptr @convert.3.clone_kernel(ptr %0) #0 {
  %convert.3.clone.invar_address.dim.1 = alloca i64, align 8
  %convert.3.clone.invar_address.dim.0 = alloca i64, align 8
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
  %lo_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @convert.3.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @convert.3.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  store i64 %lo_dim_0, ptr %convert.3.clone.invar_address.dim.0, align 4
  br label %convert.3.clone.loop_header.dim.0

convert.3.clone.loop_header.dim.0:                ; preds = %convert.3.clone.loop_exit.dim.1, %1
  %convert.3.clone.indvar.dim.0 = load i64, ptr %convert.3.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %convert.3.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %convert.3.clone.loop_exit.dim.0, label %convert.3.clone.loop_body.dim.0

convert.3.clone.loop_body.dim.0:                  ; preds = %convert.3.clone.loop_header.dim.0
  store i64 0, ptr %convert.3.clone.invar_address.dim.1, align 4
  br label %convert.3.clone.loop_header.dim.1

convert.3.clone.loop_header.dim.1:                ; preds = %convert.3.clone.loop_body.dim.1, %convert.3.clone.loop_body.dim.0
  %convert.3.clone.indvar.dim.1 = load i64, ptr %convert.3.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %convert.3.clone.indvar.dim.1, 512
  br i1 %3, label %convert.3.clone.loop_exit.dim.1, label %convert.3.clone.loop_body.dim.1

convert.3.clone.loop_body.dim.1:                  ; preds = %convert.3.clone.loop_header.dim.1
  %4 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg0, i64 0, i64 %convert.3.clone.indvar.dim.0, i64 %convert.3.clone.indvar.dim.1
  %5 = load bfloat, ptr %4, align 2, !invariant.load !1, !noalias !5
  %6 = bitcast bfloat %5 to i16
  %7 = zext i16 %6 to i32
  %8 = shl i32 %7, 16
  %9 = bitcast i32 %8 to float
  %10 = getelementptr inbounds [512 x [512 x float]], ptr %arg1, i64 0, i64 %convert.3.clone.indvar.dim.0, i64 %convert.3.clone.indvar.dim.1
  store float %9, ptr %10, align 4, !alias.scope !5
  %invar.inc3 = add nuw nsw i64 %convert.3.clone.indvar.dim.1, 1
  store i64 %invar.inc3, ptr %convert.3.clone.invar_address.dim.1, align 4
  br label %convert.3.clone.loop_header.dim.1

convert.3.clone.loop_exit.dim.1:                  ; preds = %convert.3.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %convert.3.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %convert.3.clone.invar_address.dim.0, align 4
  br label %convert.3.clone.loop_header.dim.0, !llvm.loop !8

convert.3.clone.loop_exit.dim.0:                  ; preds = %convert.3.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %convert.3.clone.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 524288}
!3 = !{i64 64}
!4 = !{i64 1048576}
!5 = !{!6}
!6 = !{!"result slice: {index:9, offset:0, size:1048576}", !7}
!7 = !{!"XLA host kernel convert.3.clone_kernel AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
