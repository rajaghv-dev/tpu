; ModuleID = '__compute_module_reduce-window.clone_elemental_kernel_module'
source_filename = "__compute_module_reduce-window.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@reduce-window.clone_parallel_bounds = private unnamed_addr constant [4 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 4]], [1 x [2 x i64]] [[2 x i64] [i64 4, i64 8]], [1 x [2 x i64]] [[2 x i64] [i64 8, i64 12]], [1 x [2 x i64]] [[2 x i64] [i64 12, i64 16]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @reduce-window.clone_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %workgroup_id_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x = load i64, ptr %workgroup_id, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %lo_dim_0_gep = getelementptr inbounds [4 x [1 x [2 x i64]]], ptr @reduce-window.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [4 x [1 x [2 x i64]]], ptr @reduce-window.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %.not17 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not17, label %reduce-window.clone.loop_header.dim.1.preheader.lr.ph, label %return

reduce-window.clone.loop_header.dim.1.preheader.lr.ph: ; preds = %1
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %2 = load float, ptr %arg1, align 64, !invariant.load !1, !noalias !6
  br label %reduce-window.clone.loop_header.dim.1.preheader

reduce-window.clone.loop_header.dim.1.preheader:  ; preds = %reduce-window.clone.loop_header.dim.1.preheader.lr.ph, %reduce-window.clone.loop_exit.dim.1
  %reduce-window.clone.invar_address.dim.0.018 = phi i64 [ %lo_dim_0, %reduce-window.clone.loop_header.dim.1.preheader.lr.ph ], [ %invar.inc, %reduce-window.clone.loop_exit.dim.1 ]
  %3 = shl nsw i64 %reduce-window.clone.invar_address.dim.0.018, 5
  br label %reduce-window.clone.loop_body.dim.1

reduce-window.clone.loop_body.dim.1:              ; preds = %reduce-window.clone.loop_header.dim.1.preheader, %reduce-window.clone.loop_exit.window.0
  %reduce-window.clone.invar_address.dim.1.016 = phi i64 [ 0, %reduce-window.clone.loop_header.dim.1.preheader ], [ %invar.inc5, %reduce-window.clone.loop_exit.window.0 ]
  %4 = shl nuw nsw i64 %reduce-window.clone.invar_address.dim.1.016, 5
  %5 = or disjoint i64 %4, 1
  %6 = or disjoint i64 %4, 2
  %7 = or disjoint i64 %4, 3
  %8 = or disjoint i64 %4, 4
  %9 = or disjoint i64 %4, 5
  %10 = or disjoint i64 %4, 6
  %11 = or disjoint i64 %4, 7
  %12 = or disjoint i64 %4, 8
  %13 = or disjoint i64 %4, 9
  %14 = or disjoint i64 %4, 10
  %15 = or disjoint i64 %4, 11
  %16 = or disjoint i64 %4, 12
  %17 = or disjoint i64 %4, 13
  %18 = or disjoint i64 %4, 14
  %19 = or disjoint i64 %4, 15
  %20 = or disjoint i64 %4, 16
  %21 = or disjoint i64 %4, 17
  %22 = or disjoint i64 %4, 18
  %23 = or disjoint i64 %4, 19
  %24 = or disjoint i64 %4, 20
  %25 = or disjoint i64 %4, 21
  %26 = or disjoint i64 %4, 22
  %27 = or disjoint i64 %4, 23
  %28 = or disjoint i64 %4, 24
  %29 = or disjoint i64 %4, 25
  %30 = or disjoint i64 %4, 26
  %31 = or disjoint i64 %4, 27
  %32 = or disjoint i64 %4, 28
  %33 = or disjoint i64 %4, 29
  %34 = or disjoint i64 %4, 30
  %35 = or disjoint i64 %4, 31
  br label %reduce-window.clone.loop_header.window.1.preheader

reduce-window.clone.loop_header.window.1.preheader: ; preds = %reduce-window.clone.loop_body.dim.1, %reduce-window.clone.loop_exit.window.1
  %reduce_window_accum_ptr.015 = phi float [ %2, %reduce-window.clone.loop_body.dim.1 ], [ %.us-phi, %reduce-window.clone.loop_exit.window.1 ]
  %reduce-window.clone.invar_address.window.0.014 = phi i64 [ 0, %reduce-window.clone.loop_body.dim.1 ], [ %invar.inc6, %reduce-window.clone.loop_exit.window.1 ]
  %36 = add nuw nsw i64 %reduce-window.clone.invar_address.window.0.014, %3
  %37 = icmp ult i64 %36, 512
  br i1 %37, label %reduce-window.clone.loop_body.window.1.us.preheader, label %reduce-window.clone.loop_exit.window.1

reduce-window.clone.loop_body.window.1.us.preheader: ; preds = %reduce-window.clone.loop_header.window.1.preheader
  %38 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %4
  %39 = load float, ptr %38, align 64, !invariant.load !1, !noalias !6
  %add.61.i.us = fadd reassoc float %reduce_window_accum_ptr.015, %39
  %40 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %5
  %41 = load float, ptr %40, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.1 = fadd reassoc float %add.61.i.us, %41
  %42 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %6
  %43 = load float, ptr %42, align 8, !invariant.load !1, !noalias !6
  %add.61.i.us.2 = fadd reassoc float %add.61.i.us.1, %43
  %44 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %7
  %45 = load float, ptr %44, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.3 = fadd reassoc float %add.61.i.us.2, %45
  %46 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %8
  %47 = load float, ptr %46, align 16, !invariant.load !1, !noalias !6
  %add.61.i.us.4 = fadd reassoc float %add.61.i.us.3, %47
  %48 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %9
  %49 = load float, ptr %48, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.5 = fadd reassoc float %add.61.i.us.4, %49
  %50 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %10
  %51 = load float, ptr %50, align 8, !invariant.load !1, !noalias !6
  %add.61.i.us.6 = fadd reassoc float %add.61.i.us.5, %51
  %52 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %11
  %53 = load float, ptr %52, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.7 = fadd reassoc float %add.61.i.us.6, %53
  %54 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %12
  %55 = load float, ptr %54, align 32, !invariant.load !1, !noalias !6
  %add.61.i.us.8 = fadd reassoc float %add.61.i.us.7, %55
  %56 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %13
  %57 = load float, ptr %56, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.9 = fadd reassoc float %add.61.i.us.8, %57
  %58 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %14
  %59 = load float, ptr %58, align 8, !invariant.load !1, !noalias !6
  %add.61.i.us.10 = fadd reassoc float %add.61.i.us.9, %59
  %60 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %15
  %61 = load float, ptr %60, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.11 = fadd reassoc float %add.61.i.us.10, %61
  %62 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %16
  %63 = load float, ptr %62, align 16, !invariant.load !1, !noalias !6
  %add.61.i.us.12 = fadd reassoc float %add.61.i.us.11, %63
  %64 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %17
  %65 = load float, ptr %64, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.13 = fadd reassoc float %add.61.i.us.12, %65
  %66 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %18
  %67 = load float, ptr %66, align 8, !invariant.load !1, !noalias !6
  %add.61.i.us.14 = fadd reassoc float %add.61.i.us.13, %67
  %68 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %19
  %69 = load float, ptr %68, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.15 = fadd reassoc float %add.61.i.us.14, %69
  %70 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %20
  %71 = load float, ptr %70, align 64, !invariant.load !1, !noalias !6
  %add.61.i.us.16 = fadd reassoc float %add.61.i.us.15, %71
  %72 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %21
  %73 = load float, ptr %72, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.17 = fadd reassoc float %add.61.i.us.16, %73
  %74 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %22
  %75 = load float, ptr %74, align 8, !invariant.load !1, !noalias !6
  %add.61.i.us.18 = fadd reassoc float %add.61.i.us.17, %75
  %76 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %23
  %77 = load float, ptr %76, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.19 = fadd reassoc float %add.61.i.us.18, %77
  %78 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %24
  %79 = load float, ptr %78, align 16, !invariant.load !1, !noalias !6
  %add.61.i.us.20 = fadd reassoc float %add.61.i.us.19, %79
  %80 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %25
  %81 = load float, ptr %80, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.21 = fadd reassoc float %add.61.i.us.20, %81
  %82 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %26
  %83 = load float, ptr %82, align 8, !invariant.load !1, !noalias !6
  %add.61.i.us.22 = fadd reassoc float %add.61.i.us.21, %83
  %84 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %27
  %85 = load float, ptr %84, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.23 = fadd reassoc float %add.61.i.us.22, %85
  %86 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %28
  %87 = load float, ptr %86, align 32, !invariant.load !1, !noalias !6
  %add.61.i.us.24 = fadd reassoc float %add.61.i.us.23, %87
  %88 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %29
  %89 = load float, ptr %88, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.25 = fadd reassoc float %add.61.i.us.24, %89
  %90 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %30
  %91 = load float, ptr %90, align 8, !invariant.load !1, !noalias !6
  %add.61.i.us.26 = fadd reassoc float %add.61.i.us.25, %91
  %92 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %31
  %93 = load float, ptr %92, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.27 = fadd reassoc float %add.61.i.us.26, %93
  %94 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %32
  %95 = load float, ptr %94, align 16, !invariant.load !1, !noalias !6
  %add.61.i.us.28 = fadd reassoc float %add.61.i.us.27, %95
  %96 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %33
  %97 = load float, ptr %96, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.29 = fadd reassoc float %add.61.i.us.28, %97
  %98 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %34
  %99 = load float, ptr %98, align 8, !invariant.load !1, !noalias !6
  %add.61.i.us.30 = fadd reassoc float %add.61.i.us.29, %99
  %100 = getelementptr inbounds nuw [512 x [512 x float]], ptr %arg0, i64 0, i64 %36, i64 %35
  %101 = load float, ptr %100, align 4, !invariant.load !1, !noalias !6
  %add.61.i.us.31 = fadd reassoc float %add.61.i.us.30, %101
  br label %reduce-window.clone.loop_exit.window.1

reduce-window.clone.loop_exit.window.1:           ; preds = %reduce-window.clone.loop_header.window.1.preheader, %reduce-window.clone.loop_body.window.1.us.preheader
  %.us-phi = phi float [ %add.61.i.us.31, %reduce-window.clone.loop_body.window.1.us.preheader ], [ %reduce_window_accum_ptr.015, %reduce-window.clone.loop_header.window.1.preheader ]
  %invar.inc6 = add nuw nsw i64 %reduce-window.clone.invar_address.window.0.014, 1
  %exitcond = icmp eq i64 %invar.inc6, 32
  br i1 %exitcond, label %reduce-window.clone.loop_exit.window.0, label %reduce-window.clone.loop_header.window.1.preheader

reduce-window.clone.loop_exit.window.0:           ; preds = %reduce-window.clone.loop_exit.window.1
  %102 = getelementptr inbounds [16 x [16 x float]], ptr %arg2, i64 0, i64 %reduce-window.clone.invar_address.dim.0.018, i64 %reduce-window.clone.invar_address.dim.1.016
  store float %.us-phi, ptr %102, align 4, !alias.scope !6
  %invar.inc5 = add nuw nsw i64 %reduce-window.clone.invar_address.dim.1.016, 1
  %exitcond20 = icmp eq i64 %invar.inc5, 16
  br i1 %exitcond20, label %reduce-window.clone.loop_exit.dim.1, label %reduce-window.clone.loop_body.dim.1

reduce-window.clone.loop_exit.dim.1:              ; preds = %reduce-window.clone.loop_exit.window.0
  %invar.inc = add nuw nsw i64 %reduce-window.clone.invar_address.dim.0.018, 1
  %exitcond21.not = icmp eq i64 %invar.inc, %up_dim_0
  br i1 %exitcond21.not, label %return, label %reduce-window.clone.loop_header.dim.1.preheader, !llvm.loop !9

return:                                           ; preds = %reduce-window.clone.loop_exit.dim.1, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 1}
!1 = !{}
!2 = !{i64 1048576}
!3 = !{i64 64}
!4 = !{i64 1024}
!5 = !{i64 4}
!6 = !{!7}
!7 = !{!"result slice: {index:9, offset:1048576, size:1024}", !8}
!8 = !{!"XLA host kernel reduce-window.clone_kernel AA domain"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
