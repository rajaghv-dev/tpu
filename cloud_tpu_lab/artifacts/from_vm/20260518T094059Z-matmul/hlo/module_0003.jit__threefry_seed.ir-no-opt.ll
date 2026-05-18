; ModuleID = '__compute_module'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@0 = private unnamed_addr constant [4 x i8] c" \00\00\00"

; Function Attrs: uwtable
define ptr @bitcast_concatenate_fusion(ptr %0) #0 {
  %bitcast_concatenate_fusion.invar_address.dim.0 = alloca i64, align 8
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
  store i64 0, ptr %bitcast_concatenate_fusion.invar_address.dim.0, align 4
  br label %bitcast_concatenate_fusion.loop_header.dim.0

bitcast_concatenate_fusion.loop_header.dim.0:     ; preds = %concatenate.0.merge, %1
  %bitcast_concatenate_fusion.indvar.dim.0 = load i64, ptr %bitcast_concatenate_fusion.invar_address.dim.0, align 4
  %2 = icmp uge i64 %bitcast_concatenate_fusion.indvar.dim.0, 2
  br i1 %2, label %bitcast_concatenate_fusion.loop_exit.dim.0, label %bitcast_concatenate_fusion.loop_body.dim.0

bitcast_concatenate_fusion.loop_body.dim.0:       ; preds = %bitcast_concatenate_fusion.loop_header.dim.0
  br label %concatenate.pivot.1.

concat_index_from_operand_id0:                    ; preds = %concatenate.pivot.0.
  %3 = phi i64 [ 0, %concatenate.pivot.0. ]
  %4 = sub nsw i64 %bitcast_concatenate_fusion.indvar.dim.0, %3
  %5 = load i32, ptr %arg0, align 4, !invariant.load !1, !noalias !5
  %constant.0 = load i32, ptr @0, align 4
  %6 = lshr i32 %5, %constant.0
  %shft.chk = icmp ult i32 %constant.0, 32
  %7 = select i1 %shft.chk, i32 %6, i32 0
  br label %concatenate.0.merge

concat_index_from_operand_id1:                    ; preds = %concatenate.pivot.1.3
  %8 = phi i64 [ 1, %concatenate.pivot.1.3 ]
  %9 = sub nsw i64 %bitcast_concatenate_fusion.indvar.dim.0, %8
  %10 = load i32, ptr %arg0, align 4, !invariant.load !1, !noalias !5
  br label %concatenate.0.merge

concatenate.pivot.1.:                             ; preds = %bitcast_concatenate_fusion.loop_body.dim.0
  %11 = icmp ult i64 %bitcast_concatenate_fusion.indvar.dim.0, 1
  br i1 %11, label %concatenate.pivot.0., label %concatenate.pivot.1.3

concatenate.pivot.0.:                             ; preds = %concatenate.pivot.1.
  br label %concat_index_from_operand_id0

concatenate.pivot.1.3:                            ; preds = %concatenate.pivot.1.
  br label %concat_index_from_operand_id1

concatenate.0.merge:                              ; preds = %concat_index_from_operand_id1, %concat_index_from_operand_id0
  %12 = phi i32 [ %7, %concat_index_from_operand_id0 ], [ %10, %concat_index_from_operand_id1 ]
  %13 = getelementptr inbounds [2 x i32], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.indvar.dim.0
  store i32 %12, ptr %13, align 4, !alias.scope !5
  %invar.inc = add nuw nsw i64 %bitcast_concatenate_fusion.indvar.dim.0, 1
  store i64 %invar.inc, ptr %bitcast_concatenate_fusion.invar_address.dim.0, align 4
  br label %bitcast_concatenate_fusion.loop_header.dim.0

bitcast_concatenate_fusion.loop_exit.dim.0:       ; preds = %bitcast_concatenate_fusion.loop_header.dim.0
  br label %return

return:                                           ; preds = %bitcast_concatenate_fusion.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 4}
!3 = !{i64 64}
!4 = !{i64 8}
!5 = !{!6}
!6 = !{!"result slice: {index:0, offset:0, size:8}", !7}
!7 = !{!"XLA host kernel bitcast_concatenate_fusion AA domain"}
