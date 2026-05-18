; ModuleID = '__compute_module_part_04'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_NumWorkGroups = type { i64, i64, i64 }
%XLA_CPU_WorkGroupId = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.10 = private unnamed_addr constant [4 x i8] c"\01\00\00\00"
@__llvmsplit_unnamed.11 = private unnamed_addr constant [4 x i8] c" \00\00\00"
@broadcast_add_fusion.clone_parallel_bounds = private constant [24 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 21]], [1 x [2 x i64]] [[2 x i64] [i64 21, i64 42]], [1 x [2 x i64]] [[2 x i64] [i64 42, i64 63]], [1 x [2 x i64]] [[2 x i64] [i64 63, i64 84]], [1 x [2 x i64]] [[2 x i64] [i64 84, i64 105]], [1 x [2 x i64]] [[2 x i64] [i64 105, i64 126]], [1 x [2 x i64]] [[2 x i64] [i64 126, i64 147]], [1 x [2 x i64]] [[2 x i64] [i64 147, i64 168]], [1 x [2 x i64]] [[2 x i64] [i64 168, i64 189]], [1 x [2 x i64]] [[2 x i64] [i64 189, i64 210]], [1 x [2 x i64]] [[2 x i64] [i64 210, i64 231]], [1 x [2 x i64]] [[2 x i64] [i64 231, i64 252]], [1 x [2 x i64]] [[2 x i64] [i64 252, i64 273]], [1 x [2 x i64]] [[2 x i64] [i64 273, i64 294]], [1 x [2 x i64]] [[2 x i64] [i64 294, i64 315]], [1 x [2 x i64]] [[2 x i64] [i64 315, i64 336]], [1 x [2 x i64]] [[2 x i64] [i64 336, i64 357]], [1 x [2 x i64]] [[2 x i64] [i64 357, i64 378]], [1 x [2 x i64]] [[2 x i64] [i64 378, i64 399]], [1 x [2 x i64]] [[2 x i64] [i64 399, i64 420]], [1 x [2 x i64]] [[2 x i64] [i64 420, i64 441]], [1 x [2 x i64]] [[2 x i64] [i64 441, i64 462]], [1 x [2 x i64]] [[2 x i64] [i64 462, i64 483]], [1 x [2 x i64]] [[2 x i64] [i64 483, i64 512]]]

; Function Attrs: uwtable
define ptr @broadcast_add_fusion.clone(ptr %0) #0 {
  %broadcast_add_fusion.clone.invar_address.dim.1 = alloca i64, align 8
  %broadcast_add_fusion.clone.invar_address.dim.0 = alloca i64, align 8
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
  %args_gep5 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args6 = load ptr, ptr %args_gep5, align 8
  %arg3_gep = getelementptr %XLA_CPU_KernelArg, ptr %args6, i32 3, i32 0
  %arg3 = load ptr, ptr %arg3_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %args_gep7 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args8 = load ptr, ptr %args_gep7, align 8
  %arg4_gep = getelementptr %XLA_CPU_KernelArg, ptr %args8, i32 4, i32 0
  %arg4 = load ptr, ptr %arg4_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep9 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args10 = load ptr, ptr %args_gep9, align 8
  %arg5_gep = getelementptr %XLA_CPU_KernelArg, ptr %args10, i32 5, i32 0
  %arg5 = load ptr, ptr %arg5_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %lo_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.clone_parallel_bounds, i32 0, i64 %workgroup_id_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  store i64 %lo_dim_0, ptr %broadcast_add_fusion.clone.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.0

broadcast_add_fusion.clone.loop_header.dim.0:     ; preds = %broadcast_add_fusion.clone.loop_exit.dim.1, %1
  %broadcast_add_fusion.clone.indvar.dim.0 = load i64, ptr %broadcast_add_fusion.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %broadcast_add_fusion.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %broadcast_add_fusion.clone.loop_exit.dim.0, label %broadcast_add_fusion.clone.loop_body.dim.0

broadcast_add_fusion.clone.loop_body.dim.0:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.0
  store i64 0, ptr %broadcast_add_fusion.clone.invar_address.dim.1, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.1

broadcast_add_fusion.clone.loop_header.dim.1:     ; preds = %broadcast_add_fusion.clone.loop_body.dim.1, %broadcast_add_fusion.clone.loop_body.dim.0
  %broadcast_add_fusion.clone.indvar.dim.1 = load i64, ptr %broadcast_add_fusion.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %broadcast_add_fusion.clone.indvar.dim.1, 512
  br i1 %3, label %broadcast_add_fusion.clone.loop_exit.dim.1, label %broadcast_add_fusion.clone.loop_body.dim.1

broadcast_add_fusion.clone.loop_body.dim.1:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.1
  %4 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1
  %5 = load i32, ptr %4, align 4, !invariant.load !1, !noalias !6
  %6 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1
  %7 = load i32, ptr %6, align 4, !invariant.load !1, !noalias !6
  %8 = add i32 %5, %7
  %9 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1
  %10 = load i32, ptr %9, align 4, !invariant.load !1, !noalias !6
  %11 = getelementptr inbounds [4 x i32], ptr %arg1, i64 0, i64 0
  %12 = load i32, ptr %11, align 4, !invariant.load !1, !noalias !6
  %13 = shl i32 %10, %12
  %shft.chk = icmp ult i32 %12, 32
  %14 = select i1 %shft.chk, i32 %13, i32 0
  %15 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1
  %16 = load i32, ptr %15, align 4, !invariant.load !1, !noalias !6
  %constant.159 = load i32, ptr @__llvmsplit_unnamed.11, align 4
  %17 = sub i32 %constant.159, %12
  %18 = lshr i32 %16, %17
  %shft.chk12 = icmp ult i32 %17, 32
  %19 = select i1 %shft.chk12, i32 %18, i32 0
  %20 = or i32 %14, %19
  %21 = xor i32 %8, %20
  %22 = add i32 %8, %21
  %23 = getelementptr inbounds [4 x i32], ptr %arg1, i64 0, i64 1
  %24 = load i32, ptr %23, align 4, !invariant.load !1, !noalias !6
  %25 = shl i32 %21, %24
  %shft.chk13 = icmp ult i32 %24, 32
  %26 = select i1 %shft.chk13, i32 %25, i32 0
  %constant.15914 = load i32, ptr @__llvmsplit_unnamed.11, align 4
  %27 = sub i32 %constant.15914, %24
  %28 = lshr i32 %21, %27
  %shft.chk15 = icmp ult i32 %27, 32
  %29 = select i1 %shft.chk15, i32 %28, i32 0
  %30 = or i32 %26, %29
  %31 = xor i32 %22, %30
  %32 = add i32 %22, %31
  %33 = getelementptr inbounds [4 x i32], ptr %arg1, i64 0, i64 2
  %34 = load i32, ptr %33, align 4, !invariant.load !1, !noalias !6
  %35 = shl i32 %31, %34
  %shft.chk16 = icmp ult i32 %34, 32
  %36 = select i1 %shft.chk16, i32 %35, i32 0
  %constant.15917 = load i32, ptr @__llvmsplit_unnamed.11, align 4
  %37 = sub i32 %constant.15917, %34
  %38 = lshr i32 %31, %37
  %shft.chk18 = icmp ult i32 %37, 32
  %39 = select i1 %shft.chk18, i32 %38, i32 0
  %40 = or i32 %36, %39
  %41 = xor i32 %32, %40
  %42 = add i32 %32, %41
  %43 = getelementptr inbounds [4 x i32], ptr %arg1, i64 0, i64 3
  %44 = load i32, ptr %43, align 4, !invariant.load !1, !noalias !6
  %45 = shl i32 %41, %44
  %shft.chk19 = icmp ult i32 %44, 32
  %46 = select i1 %shft.chk19, i32 %45, i32 0
  %constant.15920 = load i32, ptr @__llvmsplit_unnamed.11, align 4
  %47 = sub i32 %constant.15920, %44
  %48 = lshr i32 %41, %47
  %shft.chk21 = icmp ult i32 %47, 32
  %49 = select i1 %shft.chk21, i32 %48, i32 0
  %50 = or i32 %46, %49
  %51 = xor i32 %42, %50
  %52 = load i32, ptr %arg0, align 4, !invariant.load !1, !noalias !6
  %53 = add i32 %51, %52
  %54 = load i32, ptr %arg4, align 4, !invariant.load !1, !noalias !6
  %constant.160 = load i32, ptr @__llvmsplit_unnamed.10, align 4
  %55 = add i32 %54, %constant.160
  %56 = add i32 %53, %55
  %57 = getelementptr inbounds [512 x [512 x i32]], ptr %arg5, i64 0, i64 %broadcast_add_fusion.clone.indvar.dim.0, i64 %broadcast_add_fusion.clone.indvar.dim.1
  store i32 %56, ptr %57, align 4, !alias.scope !6
  %invar.inc11 = add nuw nsw i64 %broadcast_add_fusion.clone.indvar.dim.1, 1
  store i64 %invar.inc11, ptr %broadcast_add_fusion.clone.invar_address.dim.1, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.1

broadcast_add_fusion.clone.loop_exit.dim.1:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %broadcast_add_fusion.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast_add_fusion.clone.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.clone.loop_header.dim.0, !llvm.loop !9

broadcast_add_fusion.clone.loop_exit.dim.0:       ; preds = %broadcast_add_fusion.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %broadcast_add_fusion.clone.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 4}
!1 = !{}
!2 = !{i64 4}
!3 = !{i64 64}
!4 = !{i64 16}
!5 = !{i64 1048576}
!6 = !{!7}
!7 = !{!"result slice: {index:7, offset:2097216, size:1048576}", !8}
!8 = !{!"XLA host kernel broadcast_add_fusion.clone AA domain"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
