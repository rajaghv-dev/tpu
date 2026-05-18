; ModuleID = '__compute_module_part_01'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@broadcast_add_fusion.2.clone_parallel_bounds = private unnamed_addr constant [4 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 128]], [1 x [2 x i64]] [[2 x i64] [i64 128, i64 256]], [1 x [2 x i64]] [[2 x i64] [i64 256, i64 384]], [1 x [2 x i64]] [[2 x i64] [i64 384, i64 512]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast_add_fusion.2.clone(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %workgroup_id_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x = load i64, ptr %workgroup_id, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %lo_dim_0_gep = getelementptr inbounds [4 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.2.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [4 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.2.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %.not2 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not2, label %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.lr.ph, label %return

broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.lr.ph: ; preds = %1
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %2 = getelementptr inbounds nuw i8, ptr %arg0, i64 4
  %3 = load i32, ptr %2, align 4, !invariant.load !1, !noalias !5
  %broadcast.splatinsert5 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat6 = shufflevector <8 x i32> %broadcast.splatinsert5, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph

vector.ph:                                        ; preds = %broadcast_add_fusion.2.clone.loop_exit.dim.1, %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.lr.ph
  %broadcast_add_fusion.2.clone.invar_address.dim.0.03 = phi i64 [ %lo_dim_0, %broadcast_add_fusion.2.clone.loop_header.dim.1.preheader.lr.ph ], [ %invar.inc, %broadcast_add_fusion.2.clone.loop_exit.dim.1 ]
  %4 = shl i64 %broadcast_add_fusion.2.clone.invar_address.dim.0.03, 9
  %broadcast.splatinsert = insertelement <8 x i64> poison, i64 %4, i64 0
  %broadcast.splat = shufflevector <8 x i64> %broadcast.splatinsert, <8 x i64> poison, <8 x i32> zeroinitializer
  %invariant.op = add <8 x i64> splat (i64 8), %broadcast.splat
  %invariant.op7 = add <8 x i64> splat (i64 16), %broadcast.splat
  %invariant.op9 = add <8 x i64> splat (i64 24), %broadcast.splat
  %invariant.op11 = add <8 x i64> splat (i64 32), %broadcast.splat
  %invariant.op13 = add <8 x i64> splat (i64 40), %broadcast.splat
  %invariant.op15 = add <8 x i64> splat (i64 48), %broadcast.splat
  %invariant.op17 = add <8 x i64> splat (i64 56), %broadcast.splat
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  %vec.ind = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph ], [ %vec.ind.next.1, %vector.body ]
  %5 = add nuw nsw <8 x i64> %vec.ind, %broadcast.splat
  %.reass = add <8 x i64> %vec.ind, %invariant.op
  %.reass8 = add <8 x i64> %vec.ind, %invariant.op7
  %.reass10 = add <8 x i64> %vec.ind, %invariant.op9
  %6 = trunc <8 x i64> %5 to <8 x i32>
  %7 = trunc <8 x i64> %.reass to <8 x i32>
  %8 = trunc <8 x i64> %.reass8 to <8 x i32>
  %9 = trunc <8 x i64> %.reass10 to <8 x i32>
  %10 = add <8 x i32> %broadcast.splat6, %6
  %11 = add <8 x i32> %broadcast.splat6, %7
  %12 = add <8 x i32> %broadcast.splat6, %8
  %13 = add <8 x i32> %broadcast.splat6, %9
  %14 = getelementptr inbounds [512 x [512 x i32]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.2.clone.invar_address.dim.0.03, i64 %index
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 32
  %16 = getelementptr inbounds nuw i8, ptr %14, i64 64
  %17 = getelementptr inbounds nuw i8, ptr %14, i64 96
  store <8 x i32> %10, ptr %14, align 64, !alias.scope !5
  store <8 x i32> %11, ptr %15, align 32, !alias.scope !5
  store <8 x i32> %12, ptr %16, align 64, !alias.scope !5
  store <8 x i32> %13, ptr %17, align 32, !alias.scope !5
  %index.next = or disjoint i64 %index, 32
  %.reass12 = add <8 x i64> %vec.ind, %invariant.op11
  %.reass14 = add <8 x i64> %vec.ind, %invariant.op13
  %.reass16 = add <8 x i64> %vec.ind, %invariant.op15
  %.reass18 = add <8 x i64> %vec.ind, %invariant.op17
  %18 = trunc <8 x i64> %.reass12 to <8 x i32>
  %19 = trunc <8 x i64> %.reass14 to <8 x i32>
  %20 = trunc <8 x i64> %.reass16 to <8 x i32>
  %21 = trunc <8 x i64> %.reass18 to <8 x i32>
  %22 = add <8 x i32> %broadcast.splat6, %18
  %23 = add <8 x i32> %broadcast.splat6, %19
  %24 = add <8 x i32> %broadcast.splat6, %20
  %25 = add <8 x i32> %broadcast.splat6, %21
  %26 = getelementptr inbounds [512 x [512 x i32]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.2.clone.invar_address.dim.0.03, i64 %index.next
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 32
  %28 = getelementptr inbounds nuw i8, ptr %26, i64 64
  %29 = getelementptr inbounds nuw i8, ptr %26, i64 96
  store <8 x i32> %22, ptr %26, align 64, !alias.scope !5
  store <8 x i32> %23, ptr %27, align 32, !alias.scope !5
  store <8 x i32> %24, ptr %28, align 64, !alias.scope !5
  store <8 x i32> %25, ptr %29, align 32, !alias.scope !5
  %index.next.1 = add nuw nsw i64 %index, 64
  %vec.ind.next.1 = add <8 x i64> %vec.ind, splat (i64 64)
  %30 = icmp eq i64 %index.next.1, 512
  br i1 %30, label %broadcast_add_fusion.2.clone.loop_exit.dim.1, label %vector.body, !llvm.loop !8

broadcast_add_fusion.2.clone.loop_exit.dim.1:     ; preds = %vector.body
  %invar.inc = add nuw nsw i64 %broadcast_add_fusion.2.clone.invar_address.dim.0.03, 1
  %exitcond4.not = icmp eq i64 %invar.inc, %up_dim_0
  br i1 %exitcond4.not, label %return, label %vector.ph, !llvm.loop !11

return:                                           ; preds = %broadcast_add_fusion.2.clone.loop_exit.dim.1, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 1}
!1 = !{}
!2 = !{i64 1048576}
!3 = !{i64 64}
!4 = !{i64 8}
!5 = !{!6}
!6 = !{!"result slice: {index:7, offset:2097216, size:1048576}", !7}
!7 = !{!"XLA host kernel broadcast_add_fusion.2.clone AA domain"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.isvectorized", i32 1}
!10 = !{!"llvm.loop.unroll.runtime.disable"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
