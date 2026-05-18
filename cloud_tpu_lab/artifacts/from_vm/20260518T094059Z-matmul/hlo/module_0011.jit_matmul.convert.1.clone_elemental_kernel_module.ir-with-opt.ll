; ModuleID = '__compute_module_convert.1.clone_elemental_kernel_module'
source_filename = "__compute_module_convert.1.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@convert.1.clone_parallel_bounds = private unnamed_addr constant [5 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 102]], [1 x [2 x i64]] [[2 x i64] [i64 102, i64 204]], [1 x [2 x i64]] [[2 x i64] [i64 204, i64 306]], [1 x [2 x i64]] [[2 x i64] [i64 306, i64 408]], [1 x [2 x i64]] [[2 x i64] [i64 408, i64 512]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @convert.1.clone_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %workgroup_id_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x = load i64, ptr %workgroup_id, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %lo_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @convert.1.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @convert.1.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %.not5 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not5, label %vector.ph, label %return

vector.ph:                                        ; preds = %1, %convert.1.clone.loop_exit.dim.1
  %convert.1.clone.invar_address.dim.0.06 = phi i64 [ %invar.inc, %convert.1.clone.loop_exit.dim.1 ], [ %lo_dim_0, %1 ]
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next.3, %vector.body ]
  %2 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg0, i64 0, i64 %convert.1.clone.invar_address.dim.0.06, i64 %index
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 48
  %wide.load = load <8 x i16>, ptr %2, align 64, !invariant.load !1, !noalias !5
  %wide.load8 = load <8 x i16>, ptr %3, align 16, !invariant.load !1, !noalias !5
  %wide.load9 = load <8 x i16>, ptr %4, align 32, !invariant.load !1, !noalias !5
  %wide.load10 = load <8 x i16>, ptr %5, align 16, !invariant.load !1, !noalias !5
  %6 = zext <8 x i16> %wide.load to <8 x i32>
  %7 = zext <8 x i16> %wide.load8 to <8 x i32>
  %8 = zext <8 x i16> %wide.load9 to <8 x i32>
  %9 = zext <8 x i16> %wide.load10 to <8 x i32>
  %10 = shl nuw <8 x i32> %6, splat (i32 16)
  %11 = shl nuw <8 x i32> %7, splat (i32 16)
  %12 = shl nuw <8 x i32> %8, splat (i32 16)
  %13 = shl nuw <8 x i32> %9, splat (i32 16)
  %14 = getelementptr inbounds [512 x [512 x float]], ptr %arg1, i64 0, i64 %convert.1.clone.invar_address.dim.0.06, i64 %index
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 32
  %16 = getelementptr inbounds nuw i8, ptr %14, i64 64
  %17 = getelementptr inbounds nuw i8, ptr %14, i64 96
  store <8 x i32> %10, ptr %14, align 64, !alias.scope !5
  store <8 x i32> %11, ptr %15, align 32, !alias.scope !5
  store <8 x i32> %12, ptr %16, align 64, !alias.scope !5
  store <8 x i32> %13, ptr %17, align 32, !alias.scope !5
  %index.next = or disjoint i64 %index, 32
  %18 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg0, i64 0, i64 %convert.1.clone.invar_address.dim.0.06, i64 %index.next
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = getelementptr inbounds nuw i8, ptr %18, i64 32
  %21 = getelementptr inbounds nuw i8, ptr %18, i64 48
  %wide.load.1 = load <8 x i16>, ptr %18, align 64, !invariant.load !1, !noalias !5
  %wide.load8.1 = load <8 x i16>, ptr %19, align 16, !invariant.load !1, !noalias !5
  %wide.load9.1 = load <8 x i16>, ptr %20, align 32, !invariant.load !1, !noalias !5
  %wide.load10.1 = load <8 x i16>, ptr %21, align 16, !invariant.load !1, !noalias !5
  %22 = zext <8 x i16> %wide.load.1 to <8 x i32>
  %23 = zext <8 x i16> %wide.load8.1 to <8 x i32>
  %24 = zext <8 x i16> %wide.load9.1 to <8 x i32>
  %25 = zext <8 x i16> %wide.load10.1 to <8 x i32>
  %26 = shl nuw <8 x i32> %22, splat (i32 16)
  %27 = shl nuw <8 x i32> %23, splat (i32 16)
  %28 = shl nuw <8 x i32> %24, splat (i32 16)
  %29 = shl nuw <8 x i32> %25, splat (i32 16)
  %30 = getelementptr inbounds [512 x [512 x float]], ptr %arg1, i64 0, i64 %convert.1.clone.invar_address.dim.0.06, i64 %index.next
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 32
  %32 = getelementptr inbounds nuw i8, ptr %30, i64 64
  %33 = getelementptr inbounds nuw i8, ptr %30, i64 96
  store <8 x i32> %26, ptr %30, align 64, !alias.scope !5
  store <8 x i32> %27, ptr %31, align 32, !alias.scope !5
  store <8 x i32> %28, ptr %32, align 64, !alias.scope !5
  store <8 x i32> %29, ptr %33, align 32, !alias.scope !5
  %index.next.1 = or disjoint i64 %index, 64
  %34 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg0, i64 0, i64 %convert.1.clone.invar_address.dim.0.06, i64 %index.next.1
  %35 = getelementptr inbounds nuw i8, ptr %34, i64 16
  %36 = getelementptr inbounds nuw i8, ptr %34, i64 32
  %37 = getelementptr inbounds nuw i8, ptr %34, i64 48
  %wide.load.2 = load <8 x i16>, ptr %34, align 64, !invariant.load !1, !noalias !5
  %wide.load8.2 = load <8 x i16>, ptr %35, align 16, !invariant.load !1, !noalias !5
  %wide.load9.2 = load <8 x i16>, ptr %36, align 32, !invariant.load !1, !noalias !5
  %wide.load10.2 = load <8 x i16>, ptr %37, align 16, !invariant.load !1, !noalias !5
  %38 = zext <8 x i16> %wide.load.2 to <8 x i32>
  %39 = zext <8 x i16> %wide.load8.2 to <8 x i32>
  %40 = zext <8 x i16> %wide.load9.2 to <8 x i32>
  %41 = zext <8 x i16> %wide.load10.2 to <8 x i32>
  %42 = shl nuw <8 x i32> %38, splat (i32 16)
  %43 = shl nuw <8 x i32> %39, splat (i32 16)
  %44 = shl nuw <8 x i32> %40, splat (i32 16)
  %45 = shl nuw <8 x i32> %41, splat (i32 16)
  %46 = getelementptr inbounds [512 x [512 x float]], ptr %arg1, i64 0, i64 %convert.1.clone.invar_address.dim.0.06, i64 %index.next.1
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 32
  %48 = getelementptr inbounds nuw i8, ptr %46, i64 64
  %49 = getelementptr inbounds nuw i8, ptr %46, i64 96
  store <8 x i32> %42, ptr %46, align 64, !alias.scope !5
  store <8 x i32> %43, ptr %47, align 32, !alias.scope !5
  store <8 x i32> %44, ptr %48, align 64, !alias.scope !5
  store <8 x i32> %45, ptr %49, align 32, !alias.scope !5
  %index.next.2 = or disjoint i64 %index, 96
  %50 = getelementptr inbounds [512 x [512 x bfloat]], ptr %arg0, i64 0, i64 %convert.1.clone.invar_address.dim.0.06, i64 %index.next.2
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %52 = getelementptr inbounds nuw i8, ptr %50, i64 32
  %53 = getelementptr inbounds nuw i8, ptr %50, i64 48
  %wide.load.3 = load <8 x i16>, ptr %50, align 64, !invariant.load !1, !noalias !5
  %wide.load8.3 = load <8 x i16>, ptr %51, align 16, !invariant.load !1, !noalias !5
  %wide.load9.3 = load <8 x i16>, ptr %52, align 32, !invariant.load !1, !noalias !5
  %wide.load10.3 = load <8 x i16>, ptr %53, align 16, !invariant.load !1, !noalias !5
  %54 = zext <8 x i16> %wide.load.3 to <8 x i32>
  %55 = zext <8 x i16> %wide.load8.3 to <8 x i32>
  %56 = zext <8 x i16> %wide.load9.3 to <8 x i32>
  %57 = zext <8 x i16> %wide.load10.3 to <8 x i32>
  %58 = shl nuw <8 x i32> %54, splat (i32 16)
  %59 = shl nuw <8 x i32> %55, splat (i32 16)
  %60 = shl nuw <8 x i32> %56, splat (i32 16)
  %61 = shl nuw <8 x i32> %57, splat (i32 16)
  %62 = getelementptr inbounds [512 x [512 x float]], ptr %arg1, i64 0, i64 %convert.1.clone.invar_address.dim.0.06, i64 %index.next.2
  %63 = getelementptr inbounds nuw i8, ptr %62, i64 32
  %64 = getelementptr inbounds nuw i8, ptr %62, i64 64
  %65 = getelementptr inbounds nuw i8, ptr %62, i64 96
  store <8 x i32> %58, ptr %62, align 64, !alias.scope !5
  store <8 x i32> %59, ptr %63, align 32, !alias.scope !5
  store <8 x i32> %60, ptr %64, align 64, !alias.scope !5
  store <8 x i32> %61, ptr %65, align 32, !alias.scope !5
  %index.next.3 = add nuw nsw i64 %index, 128
  %66 = icmp eq i64 %index.next.3, 512
  br i1 %66, label %convert.1.clone.loop_exit.dim.1, label %vector.body, !llvm.loop !8

convert.1.clone.loop_exit.dim.1:                  ; preds = %vector.body
  %invar.inc = add nuw nsw i64 %convert.1.clone.invar_address.dim.0.06, 1
  %exitcond7.not = icmp eq i64 %invar.inc, %up_dim_0
  br i1 %exitcond7.not, label %return, label %vector.ph, !llvm.loop !11

return:                                           ; preds = %convert.1.clone.loop_exit.dim.1, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 1}
!1 = !{}
!2 = !{i64 524288}
!3 = !{i64 64}
!4 = !{i64 1048576}
!5 = !{!6}
!6 = !{!"result slice: {index:3, offset:1048576, size:1048576}", !7}
!7 = !{!"XLA host kernel convert.1.clone_kernel AA domain"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.isvectorized", i32 1}
!10 = !{!"llvm.loop.unroll.runtime.disable"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
