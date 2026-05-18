; ModuleID = '__compute_module_convert.2.clone_elemental_kernel_module'
source_filename = "__compute_module_convert.2.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@convert.2.clone_parallel_bounds = private constant [5 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 102]], [1 x [2 x i64]] [[2 x i64] [i64 102, i64 204]], [1 x [2 x i64]] [[2 x i64] [i64 204, i64 306]], [1 x [2 x i64]] [[2 x i64] [i64 306, i64 408]], [1 x [2 x i64]] [[2 x i64] [i64 408, i64 512]]]

; Function Attrs: uwtable
define ptr @convert.2.clone_kernel(ptr %0) #0 {
  %convert.2.clone.invar_address.dim.1 = alloca i64, align 8
  %convert.2.clone.invar_address.dim.0 = alloca i64, align 8
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
  %lo_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @convert.2.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @convert.2.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  store i64 %lo_dim_0, ptr %convert.2.clone.invar_address.dim.0, align 4
  br label %convert.2.clone.loop_header.dim.0

convert.2.clone.loop_header.dim.0:                ; preds = %convert.2.clone.loop_exit.dim.1, %1
  %convert.2.clone.indvar.dim.0 = load i64, ptr %convert.2.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %convert.2.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %convert.2.clone.loop_exit.dim.0, label %convert.2.clone.loop_body.dim.0

convert.2.clone.loop_body.dim.0:                  ; preds = %convert.2.clone.loop_header.dim.0
  store i64 0, ptr %convert.2.clone.invar_address.dim.1, align 4
  br label %convert.2.clone.loop_header.dim.1

convert.2.clone.loop_header.dim.1:                ; preds = %convert.2.clone.loop_body.dim.1, %convert.2.clone.loop_body.dim.0
  %convert.2.clone.indvar.dim.1 = load i64, ptr %convert.2.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %convert.2.clone.indvar.dim.1, 512
  br i1 %3, label %convert.2.clone.loop_exit.dim.1, label %convert.2.clone.loop_body.dim.1

convert.2.clone.loop_body.dim.1:                  ; preds = %convert.2.clone.loop_header.dim.1
  %4 = getelementptr inbounds [512 x [512 x float]], ptr %arg0, i64 0, i64 %convert.2.clone.indvar.dim.0, i64 %convert.2.clone.indvar.dim.1
  %5 = load float, ptr %4, align 4, !invariant.load !1, !noalias !5
  %6 = bitcast float %5 to i32
  %7 = lshr i32 %6, 16
  %8 = and i32 %7, 1
  %9 = add i32 32767, %8
  %10 = call i1 @llvm.is.fpclass.f32(float %5, i32 3)
  %11 = and i32 %6, -4194304
  %12 = or i32 %11, 4194304
  %13 = add i32 %6, %9
  %14 = select i1 %10, i32 %12, i32 %13
  %15 = lshr i32 %14, 16
  %16 = trunc i32 %15 to i16
  %17 = bitcast i16 %16 to bfloat
  %18 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg1, i64 0, i64 %convert.2.clone.indvar.dim.0, i64 %convert.2.clone.indvar.dim.1
  store bfloat %17, ptr %18, align 2, !alias.scope !5
  %invar.inc3 = add nuw nsw i64 %convert.2.clone.indvar.dim.1, 1
  store i64 %invar.inc3, ptr %convert.2.clone.invar_address.dim.1, align 4
  br label %convert.2.clone.loop_header.dim.1

convert.2.clone.loop_exit.dim.1:                  ; preds = %convert.2.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %convert.2.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %convert.2.clone.invar_address.dim.0, align 4
  br label %convert.2.clone.loop_header.dim.0, !llvm.loop !8

convert.2.clone.loop_exit.dim.0:                  ; preds = %convert.2.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %convert.2.clone.loop_exit.dim.0
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.is.fpclass.f32(float, i32 immarg) #1

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 2}
!1 = !{}
!2 = !{i64 1048576}
!3 = !{i64 64}
!4 = !{i64 524288}
!5 = !{!6}
!6 = !{!"result slice: {index:0, offset:0, size:524288}", !7}
!7 = !{!"XLA host kernel convert.2.clone_kernel AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
