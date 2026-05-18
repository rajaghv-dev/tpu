; ModuleID = '__compute_module_part_02'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.7 = private unnamed_addr constant [8 x i8] c" \00\00\00\00\00\00\00"
@__llvmsplit_unnamed.8 = private unnamed_addr constant [8 x i8] c"\00\02\00\00\00\00\00\00"
@broadcast_add_fusion.3.clone_parallel_bounds = private constant [24 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 21]], [1 x [2 x i64]] [[2 x i64] [i64 21, i64 42]], [1 x [2 x i64]] [[2 x i64] [i64 42, i64 63]], [1 x [2 x i64]] [[2 x i64] [i64 63, i64 84]], [1 x [2 x i64]] [[2 x i64] [i64 84, i64 105]], [1 x [2 x i64]] [[2 x i64] [i64 105, i64 126]], [1 x [2 x i64]] [[2 x i64] [i64 126, i64 147]], [1 x [2 x i64]] [[2 x i64] [i64 147, i64 168]], [1 x [2 x i64]] [[2 x i64] [i64 168, i64 189]], [1 x [2 x i64]] [[2 x i64] [i64 189, i64 210]], [1 x [2 x i64]] [[2 x i64] [i64 210, i64 231]], [1 x [2 x i64]] [[2 x i64] [i64 231, i64 252]], [1 x [2 x i64]] [[2 x i64] [i64 252, i64 273]], [1 x [2 x i64]] [[2 x i64] [i64 273, i64 294]], [1 x [2 x i64]] [[2 x i64] [i64 294, i64 315]], [1 x [2 x i64]] [[2 x i64] [i64 315, i64 336]], [1 x [2 x i64]] [[2 x i64] [i64 336, i64 357]], [1 x [2 x i64]] [[2 x i64] [i64 357, i64 378]], [1 x [2 x i64]] [[2 x i64] [i64 378, i64 399]], [1 x [2 x i64]] [[2 x i64] [i64 399, i64 420]], [1 x [2 x i64]] [[2 x i64] [i64 420, i64 441]], [1 x [2 x i64]] [[2 x i64] [i64 441, i64 462]], [1 x [2 x i64]] [[2 x i64] [i64 462, i64 483]], [1 x [2 x i64]] [[2 x i64] [i64 483, i64 512]]]

; Function Attrs: uwtable
define ptr @broadcast_add_fusion.3.clone(ptr %0) #0 {
  %broadcast_add_fusion.3.clone.invar_address.dim.1 = alloca i64, align 8
  %broadcast_add_fusion.3.clone.invar_address.dim.0 = alloca i64, align 8
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
  %lo_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  store i64 %lo_dim_0, ptr %broadcast_add_fusion.3.clone.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.0

broadcast_add_fusion.3.clone.loop_header.dim.0:   ; preds = %broadcast_add_fusion.3.clone.loop_exit.dim.1, %1
  %broadcast_add_fusion.3.clone.indvar.dim.0 = load i64, ptr %broadcast_add_fusion.3.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %broadcast_add_fusion.3.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %broadcast_add_fusion.3.clone.loop_exit.dim.0, label %broadcast_add_fusion.3.clone.loop_body.dim.0

broadcast_add_fusion.3.clone.loop_body.dim.0:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.0
  store i64 0, ptr %broadcast_add_fusion.3.clone.invar_address.dim.1, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.1

broadcast_add_fusion.3.clone.loop_header.dim.1:   ; preds = %broadcast_add_fusion.3.clone.loop_body.dim.1, %broadcast_add_fusion.3.clone.loop_body.dim.0
  %broadcast_add_fusion.3.clone.indvar.dim.1 = load i64, ptr %broadcast_add_fusion.3.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %broadcast_add_fusion.3.clone.indvar.dim.1, 512
  br i1 %3, label %broadcast_add_fusion.3.clone.loop_exit.dim.1, label %broadcast_add_fusion.3.clone.loop_body.dim.1

broadcast_add_fusion.3.clone.loop_body.dim.1:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.1
  %4 = mul nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.0, 1
  %5 = add nuw nsw i64 0, %4
  %constant.194 = load i64, ptr @__llvmsplit_unnamed.8, align 4
  %6 = mul i64 %5, %constant.194
  %7 = mul nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.1, 1
  %8 = add nuw nsw i64 0, %7
  %9 = add i64 %6, %8
  %constant.193 = load i64, ptr @__llvmsplit_unnamed.7, align 4
  %10 = lshr i64 %9, %constant.193
  %shft.chk = icmp ult i64 %constant.193, 64
  %11 = select i1 %shft.chk, i64 %10, i64 0
  %12 = trunc i64 %11 to i32
  %13 = getelementptr inbounds [2 x i32], ptr %arg0, i64 0, i64 0
  %14 = load i32, ptr %13, align 4, !invariant.load !1, !noalias !5
  %15 = add i32 %12, %14
  %16 = getelementptr inbounds [512 x [512 x i32]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.indvar.dim.0, i64 %broadcast_add_fusion.3.clone.indvar.dim.1
  store i32 %15, ptr %16, align 4, !alias.scope !5
  %invar.inc3 = add nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.1, 1
  store i64 %invar.inc3, ptr %broadcast_add_fusion.3.clone.invar_address.dim.1, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.1

broadcast_add_fusion.3.clone.loop_exit.dim.1:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast_add_fusion.3.clone.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.0, !llvm.loop !8

broadcast_add_fusion.3.clone.loop_exit.dim.0:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %broadcast_add_fusion.3.clone.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 2}
!1 = !{}
!2 = !{i64 8}
!3 = !{i64 64}
!4 = !{i64 1048576}
!5 = !{!6}
!6 = !{!"result slice: {index:7, offset:3145792, size:1048576}", !7}
!7 = !{!"XLA host kernel broadcast_add_fusion.3.clone AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
