; ModuleID = '__compute_module_convert.2.clone_elemental_kernel_module'
source_filename = "__compute_module_convert.2.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@convert.2.clone_parallel_bounds = private unnamed_addr constant [5 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 102]], [1 x [2 x i64]] [[2 x i64] [i64 102, i64 204]], [1 x [2 x i64]] [[2 x i64] [i64 204, i64 306]], [1 x [2 x i64]] [[2 x i64] [i64 306, i64 408]], [1 x [2 x i64]] [[2 x i64] [i64 408, i64 512]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @convert.2.clone_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %workgroup_id_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x = load i64, ptr %workgroup_id, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %lo_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @convert.2.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @convert.2.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %.not5 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not5, label %vector.ph, label %return

vector.ph:                                        ; preds = %1, %convert.2.clone.loop_exit.dim.1
  %convert.2.clone.invar_address.dim.0.06 = phi i64 [ %invar.inc, %convert.2.clone.loop_exit.dim.1 ], [ %lo_dim_0, %1 ]
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %2 = getelementptr inbounds [512 x [512 x float]], ptr %arg0, i64 0, i64 %convert.2.clone.invar_address.dim.0.06, i64 %index
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 64
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 96
  %wide.load = load <8 x float>, ptr %2, align 64, !invariant.load !1, !noalias !5
  %wide.load8 = load <8 x float>, ptr %3, align 32, !invariant.load !1, !noalias !5
  %wide.load9 = load <8 x float>, ptr %4, align 64, !invariant.load !1, !noalias !5
  %wide.load10 = load <8 x float>, ptr %5, align 32, !invariant.load !1, !noalias !5
  %6 = bitcast <8 x float> %wide.load to <8 x i32>
  %7 = bitcast <8 x float> %wide.load8 to <8 x i32>
  %8 = bitcast <8 x float> %wide.load9 to <8 x i32>
  %9 = bitcast <8 x float> %wide.load10 to <8 x i32>
  %10 = lshr <8 x i32> %6, splat (i32 16)
  %11 = lshr <8 x i32> %7, splat (i32 16)
  %12 = lshr <8 x i32> %8, splat (i32 16)
  %13 = lshr <8 x i32> %9, splat (i32 16)
  %14 = and <8 x i32> %10, splat (i32 1)
  %15 = and <8 x i32> %11, splat (i32 1)
  %16 = and <8 x i32> %12, splat (i32 1)
  %17 = and <8 x i32> %13, splat (i32 1)
  %18 = fcmp uno <8 x float> %wide.load, zeroinitializer
  %19 = fcmp uno <8 x float> %wide.load8, zeroinitializer
  %20 = fcmp uno <8 x float> %wide.load9, zeroinitializer
  %21 = fcmp uno <8 x float> %wide.load10, zeroinitializer
  %22 = and <8 x i32> %6, splat (i32 -8388608)
  %23 = and <8 x i32> %7, splat (i32 -8388608)
  %24 = and <8 x i32> %8, splat (i32 -8388608)
  %25 = and <8 x i32> %9, splat (i32 -8388608)
  %26 = or disjoint <8 x i32> %22, splat (i32 4194304)
  %27 = or disjoint <8 x i32> %23, splat (i32 4194304)
  %28 = or disjoint <8 x i32> %24, splat (i32 4194304)
  %29 = or disjoint <8 x i32> %25, splat (i32 4194304)
  %30 = add <8 x i32> %6, splat (i32 32767)
  %31 = add <8 x i32> %7, splat (i32 32767)
  %32 = add <8 x i32> %8, splat (i32 32767)
  %33 = add <8 x i32> %9, splat (i32 32767)
  %34 = add <8 x i32> %30, %14
  %35 = add <8 x i32> %31, %15
  %36 = add <8 x i32> %32, %16
  %37 = add <8 x i32> %33, %17
  %38 = select <8 x i1> %18, <8 x i32> %26, <8 x i32> %34
  %39 = select <8 x i1> %19, <8 x i32> %27, <8 x i32> %35
  %40 = select <8 x i1> %20, <8 x i32> %28, <8 x i32> %36
  %41 = select <8 x i1> %21, <8 x i32> %29, <8 x i32> %37
  %42 = lshr <8 x i32> %38, splat (i32 16)
  %43 = lshr <8 x i32> %39, splat (i32 16)
  %44 = lshr <8 x i32> %40, splat (i32 16)
  %45 = lshr <8 x i32> %41, splat (i32 16)
  %46 = trunc nuw <8 x i32> %42 to <8 x i16>
  %47 = trunc nuw <8 x i32> %43 to <8 x i16>
  %48 = trunc nuw <8 x i32> %44 to <8 x i16>
  %49 = trunc nuw <8 x i32> %45 to <8 x i16>
  %50 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg1, i64 0, i64 %convert.2.clone.invar_address.dim.0.06, i64 %index
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %52 = getelementptr inbounds nuw i8, ptr %50, i64 32
  %53 = getelementptr inbounds nuw i8, ptr %50, i64 48
  store <8 x i16> %46, ptr %50, align 64, !alias.scope !5
  store <8 x i16> %47, ptr %51, align 16, !alias.scope !5
  store <8 x i16> %48, ptr %52, align 32, !alias.scope !5
  store <8 x i16> %49, ptr %53, align 16, !alias.scope !5
  %index.next = add nuw i64 %index, 32
  %54 = icmp eq i64 %index.next, 512
  br i1 %54, label %convert.2.clone.loop_exit.dim.1, label %vector.body, !llvm.loop !8

convert.2.clone.loop_exit.dim.1:                  ; preds = %vector.body
  %invar.inc = add nuw nsw i64 %convert.2.clone.invar_address.dim.0.06, 1
  %exitcond7.not = icmp eq i64 %invar.inc, %up_dim_0
  br i1 %exitcond7.not, label %return, label %vector.ph, !llvm.loop !11

return:                                           ; preds = %convert.2.clone.loop_exit.dim.1, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 2}
!1 = !{}
!2 = !{i64 1048576}
!3 = !{i64 64}
!4 = !{i64 524288}
!5 = !{!6}
!6 = !{!"result slice: {index:0, offset:0, size:524288}", !7}
!7 = !{!"XLA host kernel convert.2.clone_kernel AA domain"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.isvectorized", i32 1}
!10 = !{!"llvm.loop.unroll.runtime.disable"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
