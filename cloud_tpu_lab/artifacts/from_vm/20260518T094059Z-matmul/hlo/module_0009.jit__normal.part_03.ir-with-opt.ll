; ModuleID = '__compute_module_part_03'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@broadcast_add_fusion.1.clone_parallel_bounds = private unnamed_addr constant [24 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 21]], [1 x [2 x i64]] [[2 x i64] [i64 21, i64 42]], [1 x [2 x i64]] [[2 x i64] [i64 42, i64 63]], [1 x [2 x i64]] [[2 x i64] [i64 63, i64 84]], [1 x [2 x i64]] [[2 x i64] [i64 84, i64 105]], [1 x [2 x i64]] [[2 x i64] [i64 105, i64 126]], [1 x [2 x i64]] [[2 x i64] [i64 126, i64 147]], [1 x [2 x i64]] [[2 x i64] [i64 147, i64 168]], [1 x [2 x i64]] [[2 x i64] [i64 168, i64 189]], [1 x [2 x i64]] [[2 x i64] [i64 189, i64 210]], [1 x [2 x i64]] [[2 x i64] [i64 210, i64 231]], [1 x [2 x i64]] [[2 x i64] [i64 231, i64 252]], [1 x [2 x i64]] [[2 x i64] [i64 252, i64 273]], [1 x [2 x i64]] [[2 x i64] [i64 273, i64 294]], [1 x [2 x i64]] [[2 x i64] [i64 294, i64 315]], [1 x [2 x i64]] [[2 x i64] [i64 315, i64 336]], [1 x [2 x i64]] [[2 x i64] [i64 336, i64 357]], [1 x [2 x i64]] [[2 x i64] [i64 357, i64 378]], [1 x [2 x i64]] [[2 x i64] [i64 378, i64 399]], [1 x [2 x i64]] [[2 x i64] [i64 399, i64 420]], [1 x [2 x i64]] [[2 x i64] [i64 420, i64 441]], [1 x [2 x i64]] [[2 x i64] [i64 441, i64 462]], [1 x [2 x i64]] [[2 x i64] [i64 462, i64 483]], [1 x [2 x i64]] [[2 x i64] [i64 483, i64 512]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast_add_fusion.1.clone(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %workgroup_id_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %workgroup_id = load ptr, ptr %workgroup_id_gep, align 8
  %workgroup_id_x = load i64, ptr %workgroup_id, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg3_gep = getelementptr i8, ptr %args, i64 48
  %arg3 = load ptr, ptr %arg3_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg4_gep = getelementptr i8, ptr %args, i64 64
  %arg4 = load ptr, ptr %arg4_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %lo_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.1.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [24 x [1 x [2 x i64]]], ptr @broadcast_add_fusion.1.clone_parallel_bounds, i64 0, i64 %workgroup_id_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %.not2 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not2, label %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph, label %return

broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph: ; preds = %1
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %2 = load i32, ptr %arg1, align 64, !invariant.load !1, !noalias !6
  %.fr = freeze i32 %2
  %shft.chk = icmp ult i32 %.fr, 32
  %3 = sub i32 32, %.fr
  %shft.chk10 = icmp ult i32 %3, 32
  %4 = getelementptr inbounds nuw i8, ptr %arg1, i64 4
  %5 = load i32, ptr %4, align 4, !invariant.load !1, !noalias !6
  %.fr4 = freeze i32 %5
  %shft.chk11 = icmp ult i32 %.fr4, 32
  %6 = sub i32 32, %.fr4
  %shft.chk13 = icmp ult i32 %6, 32
  %7 = getelementptr inbounds nuw i8, ptr %arg1, i64 8
  %8 = load i32, ptr %7, align 8, !invariant.load !1, !noalias !6
  %.fr5 = freeze i32 %8
  %shft.chk14 = icmp ult i32 %.fr5, 32
  %9 = sub i32 32, %.fr5
  %shft.chk16 = icmp ult i32 %9, 32
  %10 = load i32, ptr %arg0, align 64, !invariant.load !1, !noalias !6
  br i1 %shft.chk, label %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us, label %vector.ph.preheader

vector.ph.preheader:                              ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph
  %broadcast.splatinsert = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert37 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat38 = shufflevector <8 x i32> %broadcast.splatinsert37, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert39 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat40 = shufflevector <8 x i32> %broadcast.splatinsert39, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert41 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat42 = shufflevector <8 x i32> %broadcast.splatinsert41, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert43 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat44 = shufflevector <8 x i32> %broadcast.splatinsert43, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert45 = insertelement <8 x i32> poison, i32 %10, i64 0
  %broadcast.splat46 = shufflevector <8 x i32> %broadcast.splatinsert45, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph

broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us: ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph
  br i1 %shft.chk10, label %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us, label %vector.ph49.preheader

vector.ph49.preheader:                            ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us
  %broadcast.splatinsert50 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat51 = shufflevector <8 x i32> %broadcast.splatinsert50, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert52 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat53 = shufflevector <8 x i32> %broadcast.splatinsert52, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert54 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat55 = shufflevector <8 x i32> %broadcast.splatinsert54, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert56 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat57 = shufflevector <8 x i32> %broadcast.splatinsert56, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert58 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat59 = shufflevector <8 x i32> %broadcast.splatinsert58, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert60 = insertelement <8 x i32> poison, i32 %10, i64 0
  %broadcast.splat61 = shufflevector <8 x i32> %broadcast.splatinsert60, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph49

broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us: ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us
  br i1 %shft.chk11, label %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us, label %vector.ph70.preheader

vector.ph70.preheader:                            ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us
  %broadcast.splatinsert71 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat72 = shufflevector <8 x i32> %broadcast.splatinsert71, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert73 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat74 = shufflevector <8 x i32> %broadcast.splatinsert73, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert75 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat76 = shufflevector <8 x i32> %broadcast.splatinsert75, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert77 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat78 = shufflevector <8 x i32> %broadcast.splatinsert77, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert79 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat80 = shufflevector <8 x i32> %broadcast.splatinsert79, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert81 = insertelement <8 x i32> poison, i32 %10, i64 0
  %broadcast.splat82 = shufflevector <8 x i32> %broadcast.splatinsert81, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph70

broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us: ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us
  br i1 %shft.chk13, label %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us, label %vector.ph91.preheader

vector.ph91.preheader:                            ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us
  %broadcast.splatinsert92 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat93 = shufflevector <8 x i32> %broadcast.splatinsert92, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert94 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat95 = shufflevector <8 x i32> %broadcast.splatinsert94, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert96 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat97 = shufflevector <8 x i32> %broadcast.splatinsert96, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert98 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat99 = shufflevector <8 x i32> %broadcast.splatinsert98, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert100 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat101 = shufflevector <8 x i32> %broadcast.splatinsert100, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert102 = insertelement <8 x i32> poison, i32 %10, i64 0
  %broadcast.splat103 = shufflevector <8 x i32> %broadcast.splatinsert102, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph91

broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us: ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us
  br i1 %shft.chk14, label %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us, label %vector.ph112.preheader

vector.ph112.preheader:                           ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us
  %broadcast.splatinsert113 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat114 = shufflevector <8 x i32> %broadcast.splatinsert113, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert115 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat116 = shufflevector <8 x i32> %broadcast.splatinsert115, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert117 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat118 = shufflevector <8 x i32> %broadcast.splatinsert117, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert119 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat120 = shufflevector <8 x i32> %broadcast.splatinsert119, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert121 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat122 = shufflevector <8 x i32> %broadcast.splatinsert121, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert123 = insertelement <8 x i32> poison, i32 %10, i64 0
  %broadcast.splat124 = shufflevector <8 x i32> %broadcast.splatinsert123, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph112

broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us: ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us
  br i1 %shft.chk16, label %vector.ph154.preheader, label %vector.ph133.preheader

vector.ph133.preheader:                           ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us
  %broadcast.splatinsert134 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat135 = shufflevector <8 x i32> %broadcast.splatinsert134, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert136 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat137 = shufflevector <8 x i32> %broadcast.splatinsert136, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert138 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat139 = shufflevector <8 x i32> %broadcast.splatinsert138, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert140 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat141 = shufflevector <8 x i32> %broadcast.splatinsert140, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert142 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat143 = shufflevector <8 x i32> %broadcast.splatinsert142, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert144 = insertelement <8 x i32> poison, i32 %10, i64 0
  %broadcast.splat145 = shufflevector <8 x i32> %broadcast.splatinsert144, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph133

vector.ph154.preheader:                           ; preds = %broadcast_add_fusion.1.clone.loop_header.dim.1.preheader.lr.ph.split.us.split.us.split.us.split.us.split.us
  %broadcast.splatinsert155 = insertelement <8 x i32> poison, i32 %.fr, i64 0
  %broadcast.splat156 = shufflevector <8 x i32> %broadcast.splatinsert155, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert157 = insertelement <8 x i32> poison, i32 %3, i64 0
  %broadcast.splat158 = shufflevector <8 x i32> %broadcast.splatinsert157, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert159 = insertelement <8 x i32> poison, i32 %.fr4, i64 0
  %broadcast.splat160 = shufflevector <8 x i32> %broadcast.splatinsert159, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert161 = insertelement <8 x i32> poison, i32 %6, i64 0
  %broadcast.splat162 = shufflevector <8 x i32> %broadcast.splatinsert161, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert163 = insertelement <8 x i32> poison, i32 %.fr5, i64 0
  %broadcast.splat164 = shufflevector <8 x i32> %broadcast.splatinsert163, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert165 = insertelement <8 x i32> poison, i32 %9, i64 0
  %broadcast.splat166 = shufflevector <8 x i32> %broadcast.splatinsert165, <8 x i32> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert167 = insertelement <8 x i32> poison, i32 %10, i64 0
  %broadcast.splat168 = shufflevector <8 x i32> %broadcast.splatinsert167, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vector.ph154

vector.ph154:                                     ; preds = %vector.ph154.preheader, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us
  %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us.us = phi i64 [ %invar.inc.us.us.us.us.us.us, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us ], [ %lo_dim_0, %vector.ph154.preheader ]
  br label %vector.body169

vector.body169:                                   ; preds = %vector.body169, %vector.ph154
  %index170 = phi i64 [ 0, %vector.ph154 ], [ %index.next173.1, %vector.body169 ]
  %11 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index170
  %wide.load171 = load <8 x i32>, ptr %11, align 64, !invariant.load !1, !noalias !6
  %12 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index170
  %wide.load172 = load <8 x i32>, ptr %12, align 64, !invariant.load !1, !noalias !6
  %13 = add <8 x i32> %wide.load172, %wide.load171
  %14 = shl <8 x i32> %wide.load172, %broadcast.splat156
  %15 = lshr <8 x i32> %wide.load172, %broadcast.splat158
  %16 = or disjoint <8 x i32> %15, %14
  %17 = xor <8 x i32> %16, %13
  %18 = add <8 x i32> %17, %13
  %19 = shl <8 x i32> %17, %broadcast.splat160
  %20 = lshr <8 x i32> %17, %broadcast.splat162
  %21 = or disjoint <8 x i32> %19, %20
  %22 = xor <8 x i32> %21, %18
  %23 = add <8 x i32> %22, %18
  %24 = shl <8 x i32> %22, %broadcast.splat164
  %25 = lshr <8 x i32> %22, %broadcast.splat166
  %26 = or disjoint <8 x i32> %24, %25
  %27 = xor <8 x i32> %26, %23
  %28 = add <8 x i32> %23, %broadcast.splat168
  %29 = add <8 x i32> %28, %27
  %30 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index170
  store <8 x i32> %29, ptr %30, align 64, !alias.scope !6
  %index.next173 = or disjoint i64 %index170, 8
  %31 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index.next173
  %wide.load171.1 = load <8 x i32>, ptr %31, align 32, !invariant.load !1, !noalias !6
  %32 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index.next173
  %wide.load172.1 = load <8 x i32>, ptr %32, align 32, !invariant.load !1, !noalias !6
  %33 = add <8 x i32> %wide.load172.1, %wide.load171.1
  %34 = shl <8 x i32> %wide.load172.1, %broadcast.splat156
  %35 = lshr <8 x i32> %wide.load172.1, %broadcast.splat158
  %36 = or disjoint <8 x i32> %35, %34
  %37 = xor <8 x i32> %36, %33
  %38 = add <8 x i32> %37, %33
  %39 = shl <8 x i32> %37, %broadcast.splat160
  %40 = lshr <8 x i32> %37, %broadcast.splat162
  %41 = or disjoint <8 x i32> %39, %40
  %42 = xor <8 x i32> %41, %38
  %43 = add <8 x i32> %42, %38
  %44 = shl <8 x i32> %42, %broadcast.splat164
  %45 = lshr <8 x i32> %42, %broadcast.splat166
  %46 = or disjoint <8 x i32> %44, %45
  %47 = xor <8 x i32> %46, %43
  %48 = add <8 x i32> %43, %broadcast.splat168
  %49 = add <8 x i32> %48, %47
  %50 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us.us, i64 %index.next173
  store <8 x i32> %49, ptr %50, align 32, !alias.scope !6
  %index.next173.1 = add nuw nsw i64 %index170, 16
  %51 = icmp eq i64 %index.next173.1, 512
  br i1 %51, label %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us, label %vector.body169, !llvm.loop !9

broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us: ; preds = %vector.body169
  %invar.inc.us.us.us.us.us.us = add nuw nsw i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us.us, 1
  %exitcond24.not = icmp eq i64 %invar.inc.us.us.us.us.us.us, %up_dim_0
  br i1 %exitcond24.not, label %return, label %vector.ph154, !llvm.loop !12

vector.ph133:                                     ; preds = %vector.ph133.preheader, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split
  %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us = phi i64 [ %invar.inc.us.us.us.us.us, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph133.preheader ]
  br label %vector.body146

vector.body146:                                   ; preds = %vector.body146, %vector.ph133
  %index147 = phi i64 [ 0, %vector.ph133 ], [ %index.next150.1, %vector.body146 ]
  %52 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index147
  %wide.load148 = load <8 x i32>, ptr %52, align 64, !invariant.load !1, !noalias !6
  %53 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index147
  %wide.load149 = load <8 x i32>, ptr %53, align 64, !invariant.load !1, !noalias !6
  %54 = add <8 x i32> %wide.load149, %wide.load148
  %55 = shl <8 x i32> %wide.load149, %broadcast.splat135
  %56 = lshr <8 x i32> %wide.load149, %broadcast.splat137
  %57 = or disjoint <8 x i32> %56, %55
  %58 = xor <8 x i32> %57, %54
  %59 = add <8 x i32> %58, %54
  %60 = shl <8 x i32> %58, %broadcast.splat139
  %61 = lshr <8 x i32> %58, %broadcast.splat141
  %62 = or disjoint <8 x i32> %60, %61
  %63 = xor <8 x i32> %62, %59
  %64 = add <8 x i32> %63, %59
  %65 = shl <8 x i32> %63, %broadcast.splat143
  %66 = xor <8 x i32> %65, %64
  %67 = add <8 x i32> %64, %broadcast.splat145
  %68 = add <8 x i32> %67, %66
  %69 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index147
  store <8 x i32> %68, ptr %69, align 64, !alias.scope !6
  %index.next150 = or disjoint i64 %index147, 8
  %70 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index.next150
  %wide.load148.1 = load <8 x i32>, ptr %70, align 32, !invariant.load !1, !noalias !6
  %71 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index.next150
  %wide.load149.1 = load <8 x i32>, ptr %71, align 32, !invariant.load !1, !noalias !6
  %72 = add <8 x i32> %wide.load149.1, %wide.load148.1
  %73 = shl <8 x i32> %wide.load149.1, %broadcast.splat135
  %74 = lshr <8 x i32> %wide.load149.1, %broadcast.splat137
  %75 = or disjoint <8 x i32> %74, %73
  %76 = xor <8 x i32> %75, %72
  %77 = add <8 x i32> %76, %72
  %78 = shl <8 x i32> %76, %broadcast.splat139
  %79 = lshr <8 x i32> %76, %broadcast.splat141
  %80 = or disjoint <8 x i32> %78, %79
  %81 = xor <8 x i32> %80, %77
  %82 = add <8 x i32> %81, %77
  %83 = shl <8 x i32> %81, %broadcast.splat143
  %84 = xor <8 x i32> %83, %82
  %85 = add <8 x i32> %82, %broadcast.splat145
  %86 = add <8 x i32> %85, %84
  %87 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us, i64 %index.next150
  store <8 x i32> %86, ptr %87, align 32, !alias.scope !6
  %index.next150.1 = add nuw nsw i64 %index147, 16
  %88 = icmp eq i64 %index.next150.1, 512
  br i1 %88, label %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split, label %vector.body146, !llvm.loop !14

broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split: ; preds = %vector.body146
  %invar.inc.us.us.us.us.us = add nuw nsw i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us.us, 1
  %exitcond22.not = icmp eq i64 %invar.inc.us.us.us.us.us, %up_dim_0
  br i1 %exitcond22.not, label %return, label %vector.ph133, !llvm.loop !12

vector.ph112:                                     ; preds = %vector.ph112.preheader, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split
  %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us = phi i64 [ %invar.inc.us.us.us.us, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph112.preheader ]
  br label %vector.body125

vector.body125:                                   ; preds = %vector.body125, %vector.ph112
  %index126 = phi i64 [ 0, %vector.ph112 ], [ %index.next129.1, %vector.body125 ]
  %89 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us, i64 %index126
  %wide.load127 = load <8 x i32>, ptr %89, align 64, !invariant.load !1, !noalias !6
  %90 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us, i64 %index126
  %wide.load128 = load <8 x i32>, ptr %90, align 64, !invariant.load !1, !noalias !6
  %91 = add <8 x i32> %wide.load128, %wide.load127
  %92 = shl <8 x i32> %wide.load128, %broadcast.splat114
  %93 = lshr <8 x i32> %wide.load128, %broadcast.splat116
  %94 = or disjoint <8 x i32> %93, %92
  %95 = xor <8 x i32> %94, %91
  %96 = add <8 x i32> %95, %91
  %97 = shl <8 x i32> %95, %broadcast.splat118
  %98 = lshr <8 x i32> %95, %broadcast.splat120
  %99 = or disjoint <8 x i32> %97, %98
  %100 = xor <8 x i32> %99, %96
  %101 = add <8 x i32> %100, %96
  %102 = lshr <8 x i32> %100, %broadcast.splat122
  %103 = select i1 %shft.chk16, <8 x i32> %102, <8 x i32> zeroinitializer
  %104 = xor <8 x i32> %103, %101
  %105 = add <8 x i32> %101, %broadcast.splat124
  %106 = add <8 x i32> %105, %104
  %107 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us, i64 %index126
  store <8 x i32> %106, ptr %107, align 64, !alias.scope !6
  %index.next129 = or disjoint i64 %index126, 8
  %108 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us, i64 %index.next129
  %wide.load127.1 = load <8 x i32>, ptr %108, align 32, !invariant.load !1, !noalias !6
  %109 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us, i64 %index.next129
  %wide.load128.1 = load <8 x i32>, ptr %109, align 32, !invariant.load !1, !noalias !6
  %110 = add <8 x i32> %wide.load128.1, %wide.load127.1
  %111 = shl <8 x i32> %wide.load128.1, %broadcast.splat114
  %112 = lshr <8 x i32> %wide.load128.1, %broadcast.splat116
  %113 = or disjoint <8 x i32> %112, %111
  %114 = xor <8 x i32> %113, %110
  %115 = add <8 x i32> %114, %110
  %116 = shl <8 x i32> %114, %broadcast.splat118
  %117 = lshr <8 x i32> %114, %broadcast.splat120
  %118 = or disjoint <8 x i32> %116, %117
  %119 = xor <8 x i32> %118, %115
  %120 = add <8 x i32> %119, %115
  %121 = lshr <8 x i32> %119, %broadcast.splat122
  %122 = select i1 %shft.chk16, <8 x i32> %121, <8 x i32> zeroinitializer
  %123 = xor <8 x i32> %122, %120
  %124 = add <8 x i32> %120, %broadcast.splat124
  %125 = add <8 x i32> %124, %123
  %126 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us, i64 %index.next129
  store <8 x i32> %125, ptr %126, align 32, !alias.scope !6
  %index.next129.1 = add nuw nsw i64 %index126, 16
  %127 = icmp eq i64 %index.next129.1, 512
  br i1 %127, label %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split, label %vector.body125, !llvm.loop !15

broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split: ; preds = %vector.body125
  %invar.inc.us.us.us.us = add nuw nsw i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us.us, 1
  %exitcond20.not = icmp eq i64 %invar.inc.us.us.us.us, %up_dim_0
  br i1 %exitcond20.not, label %return, label %vector.ph112, !llvm.loop !12

vector.ph91:                                      ; preds = %vector.ph91.preheader, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split
  %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us = phi i64 [ %invar.inc.us.us.us, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph91.preheader ]
  br label %vector.body104

vector.body104:                                   ; preds = %vector.body104, %vector.ph91
  %index105 = phi i64 [ 0, %vector.ph91 ], [ %index.next108.1, %vector.body104 ]
  %128 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us, i64 %index105
  %wide.load106 = load <8 x i32>, ptr %128, align 64, !invariant.load !1, !noalias !6
  %129 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us, i64 %index105
  %wide.load107 = load <8 x i32>, ptr %129, align 64, !invariant.load !1, !noalias !6
  %130 = add <8 x i32> %wide.load107, %wide.load106
  %131 = shl <8 x i32> %wide.load107, %broadcast.splat93
  %132 = lshr <8 x i32> %wide.load107, %broadcast.splat95
  %133 = or disjoint <8 x i32> %132, %131
  %134 = xor <8 x i32> %133, %130
  %135 = add <8 x i32> %134, %130
  %136 = shl <8 x i32> %134, %broadcast.splat97
  %137 = xor <8 x i32> %136, %135
  %138 = add <8 x i32> %137, %135
  %139 = shl <8 x i32> %137, %broadcast.splat99
  %140 = select i1 %shft.chk14, <8 x i32> %139, <8 x i32> zeroinitializer
  %141 = lshr <8 x i32> %137, %broadcast.splat101
  %142 = select i1 %shft.chk16, <8 x i32> %141, <8 x i32> zeroinitializer
  %143 = or <8 x i32> %140, %142
  %144 = xor <8 x i32> %143, %138
  %145 = add <8 x i32> %138, %broadcast.splat103
  %146 = add <8 x i32> %145, %144
  %147 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us, i64 %index105
  store <8 x i32> %146, ptr %147, align 64, !alias.scope !6
  %index.next108 = or disjoint i64 %index105, 8
  %148 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us, i64 %index.next108
  %wide.load106.1 = load <8 x i32>, ptr %148, align 32, !invariant.load !1, !noalias !6
  %149 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us, i64 %index.next108
  %wide.load107.1 = load <8 x i32>, ptr %149, align 32, !invariant.load !1, !noalias !6
  %150 = add <8 x i32> %wide.load107.1, %wide.load106.1
  %151 = shl <8 x i32> %wide.load107.1, %broadcast.splat93
  %152 = lshr <8 x i32> %wide.load107.1, %broadcast.splat95
  %153 = or disjoint <8 x i32> %152, %151
  %154 = xor <8 x i32> %153, %150
  %155 = add <8 x i32> %154, %150
  %156 = shl <8 x i32> %154, %broadcast.splat97
  %157 = xor <8 x i32> %156, %155
  %158 = add <8 x i32> %157, %155
  %159 = shl <8 x i32> %157, %broadcast.splat99
  %160 = select i1 %shft.chk14, <8 x i32> %159, <8 x i32> zeroinitializer
  %161 = lshr <8 x i32> %157, %broadcast.splat101
  %162 = select i1 %shft.chk16, <8 x i32> %161, <8 x i32> zeroinitializer
  %163 = or <8 x i32> %160, %162
  %164 = xor <8 x i32> %163, %158
  %165 = add <8 x i32> %158, %broadcast.splat103
  %166 = add <8 x i32> %165, %164
  %167 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us, i64 %index.next108
  store <8 x i32> %166, ptr %167, align 32, !alias.scope !6
  %index.next108.1 = add nuw nsw i64 %index105, 16
  %168 = icmp eq i64 %index.next108.1, 512
  br i1 %168, label %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split, label %vector.body104, !llvm.loop !16

broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split: ; preds = %vector.body104
  %invar.inc.us.us.us = add nuw nsw i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us.us, 1
  %exitcond18.not = icmp eq i64 %invar.inc.us.us.us, %up_dim_0
  br i1 %exitcond18.not, label %return, label %vector.ph91, !llvm.loop !12

vector.ph70:                                      ; preds = %vector.ph70.preheader, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split
  %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us = phi i64 [ %invar.inc.us.us, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split ], [ %lo_dim_0, %vector.ph70.preheader ]
  br label %vector.body83

vector.body83:                                    ; preds = %vector.body83, %vector.ph70
  %index84 = phi i64 [ 0, %vector.ph70 ], [ %index.next87.1, %vector.body83 ]
  %169 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us, i64 %index84
  %wide.load85 = load <8 x i32>, ptr %169, align 64, !invariant.load !1, !noalias !6
  %170 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us, i64 %index84
  %wide.load86 = load <8 x i32>, ptr %170, align 64, !invariant.load !1, !noalias !6
  %171 = add <8 x i32> %wide.load86, %wide.load85
  %172 = shl <8 x i32> %wide.load86, %broadcast.splat72
  %173 = lshr <8 x i32> %wide.load86, %broadcast.splat74
  %174 = or disjoint <8 x i32> %173, %172
  %175 = xor <8 x i32> %174, %171
  %176 = add <8 x i32> %175, %171
  %177 = lshr <8 x i32> %175, %broadcast.splat76
  %178 = select i1 %shft.chk13, <8 x i32> %177, <8 x i32> zeroinitializer
  %179 = xor <8 x i32> %178, %176
  %180 = add <8 x i32> %179, %176
  %181 = shl <8 x i32> %179, %broadcast.splat78
  %182 = select i1 %shft.chk14, <8 x i32> %181, <8 x i32> zeroinitializer
  %183 = lshr <8 x i32> %179, %broadcast.splat80
  %184 = select i1 %shft.chk16, <8 x i32> %183, <8 x i32> zeroinitializer
  %185 = or <8 x i32> %182, %184
  %186 = xor <8 x i32> %185, %180
  %187 = add <8 x i32> %180, %broadcast.splat82
  %188 = add <8 x i32> %187, %186
  %189 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us, i64 %index84
  store <8 x i32> %188, ptr %189, align 64, !alias.scope !6
  %index.next87 = or disjoint i64 %index84, 8
  %190 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us, i64 %index.next87
  %wide.load85.1 = load <8 x i32>, ptr %190, align 32, !invariant.load !1, !noalias !6
  %191 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us, i64 %index.next87
  %wide.load86.1 = load <8 x i32>, ptr %191, align 32, !invariant.load !1, !noalias !6
  %192 = add <8 x i32> %wide.load86.1, %wide.load85.1
  %193 = shl <8 x i32> %wide.load86.1, %broadcast.splat72
  %194 = lshr <8 x i32> %wide.load86.1, %broadcast.splat74
  %195 = or disjoint <8 x i32> %194, %193
  %196 = xor <8 x i32> %195, %192
  %197 = add <8 x i32> %196, %192
  %198 = lshr <8 x i32> %196, %broadcast.splat76
  %199 = select i1 %shft.chk13, <8 x i32> %198, <8 x i32> zeroinitializer
  %200 = xor <8 x i32> %199, %197
  %201 = add <8 x i32> %200, %197
  %202 = shl <8 x i32> %200, %broadcast.splat78
  %203 = select i1 %shft.chk14, <8 x i32> %202, <8 x i32> zeroinitializer
  %204 = lshr <8 x i32> %200, %broadcast.splat80
  %205 = select i1 %shft.chk16, <8 x i32> %204, <8 x i32> zeroinitializer
  %206 = or <8 x i32> %203, %205
  %207 = xor <8 x i32> %206, %201
  %208 = add <8 x i32> %201, %broadcast.splat82
  %209 = add <8 x i32> %208, %207
  %210 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us, i64 %index.next87
  store <8 x i32> %209, ptr %210, align 32, !alias.scope !6
  %index.next87.1 = add nuw nsw i64 %index84, 16
  %211 = icmp eq i64 %index.next87.1, 512
  br i1 %211, label %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split, label %vector.body83, !llvm.loop !17

broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split: ; preds = %vector.body83
  %invar.inc.us.us = add nuw nsw i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us.us, 1
  %exitcond16.not = icmp eq i64 %invar.inc.us.us, %up_dim_0
  br i1 %exitcond16.not, label %return, label %vector.ph70, !llvm.loop !12

vector.ph49:                                      ; preds = %vector.ph49.preheader, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split
  %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us = phi i64 [ %invar.inc.us, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split ], [ %lo_dim_0, %vector.ph49.preheader ]
  br label %vector.body62

vector.body62:                                    ; preds = %vector.body62, %vector.ph49
  %index63 = phi i64 [ 0, %vector.ph49 ], [ %index.next66.1, %vector.body62 ]
  %212 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us, i64 %index63
  %wide.load64 = load <8 x i32>, ptr %212, align 64, !invariant.load !1, !noalias !6
  %213 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us, i64 %index63
  %wide.load65 = load <8 x i32>, ptr %213, align 64, !invariant.load !1, !noalias !6
  %214 = add <8 x i32> %wide.load65, %wide.load64
  %215 = shl <8 x i32> %wide.load65, %broadcast.splat51
  %216 = xor <8 x i32> %215, %214
  %217 = add <8 x i32> %216, %214
  %218 = shl <8 x i32> %216, %broadcast.splat53
  %219 = select i1 %shft.chk11, <8 x i32> %218, <8 x i32> zeroinitializer
  %220 = lshr <8 x i32> %216, %broadcast.splat55
  %221 = select i1 %shft.chk13, <8 x i32> %220, <8 x i32> zeroinitializer
  %222 = or <8 x i32> %219, %221
  %223 = xor <8 x i32> %222, %217
  %224 = add <8 x i32> %223, %217
  %225 = shl <8 x i32> %223, %broadcast.splat57
  %226 = select i1 %shft.chk14, <8 x i32> %225, <8 x i32> zeroinitializer
  %227 = lshr <8 x i32> %223, %broadcast.splat59
  %228 = select i1 %shft.chk16, <8 x i32> %227, <8 x i32> zeroinitializer
  %229 = or <8 x i32> %226, %228
  %230 = xor <8 x i32> %229, %224
  %231 = add <8 x i32> %224, %broadcast.splat61
  %232 = add <8 x i32> %231, %230
  %233 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us, i64 %index63
  store <8 x i32> %232, ptr %233, align 64, !alias.scope !6
  %index.next66 = or disjoint i64 %index63, 8
  %234 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us, i64 %index.next66
  %wide.load64.1 = load <8 x i32>, ptr %234, align 32, !invariant.load !1, !noalias !6
  %235 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us, i64 %index.next66
  %wide.load65.1 = load <8 x i32>, ptr %235, align 32, !invariant.load !1, !noalias !6
  %236 = add <8 x i32> %wide.load65.1, %wide.load64.1
  %237 = shl <8 x i32> %wide.load65.1, %broadcast.splat51
  %238 = xor <8 x i32> %237, %236
  %239 = add <8 x i32> %238, %236
  %240 = shl <8 x i32> %238, %broadcast.splat53
  %241 = select i1 %shft.chk11, <8 x i32> %240, <8 x i32> zeroinitializer
  %242 = lshr <8 x i32> %238, %broadcast.splat55
  %243 = select i1 %shft.chk13, <8 x i32> %242, <8 x i32> zeroinitializer
  %244 = or <8 x i32> %241, %243
  %245 = xor <8 x i32> %244, %239
  %246 = add <8 x i32> %245, %239
  %247 = shl <8 x i32> %245, %broadcast.splat57
  %248 = select i1 %shft.chk14, <8 x i32> %247, <8 x i32> zeroinitializer
  %249 = lshr <8 x i32> %245, %broadcast.splat59
  %250 = select i1 %shft.chk16, <8 x i32> %249, <8 x i32> zeroinitializer
  %251 = or <8 x i32> %248, %250
  %252 = xor <8 x i32> %251, %246
  %253 = add <8 x i32> %246, %broadcast.splat61
  %254 = add <8 x i32> %253, %252
  %255 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us, i64 %index.next66
  store <8 x i32> %254, ptr %255, align 32, !alias.scope !6
  %index.next66.1 = add nuw nsw i64 %index63, 16
  %256 = icmp eq i64 %index.next66.1, 512
  br i1 %256, label %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split, label %vector.body62, !llvm.loop !18

broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split: ; preds = %vector.body62
  %invar.inc.us = add nuw nsw i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03.us, 1
  %exitcond14.not = icmp eq i64 %invar.inc.us, %up_dim_0
  br i1 %exitcond14.not, label %return, label %vector.ph49, !llvm.loop !12

vector.ph:                                        ; preds = %vector.ph.preheader, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split
  %broadcast_add_fusion.1.clone.invar_address.dim.0.03 = phi i64 [ %invar.inc, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split ], [ %lo_dim_0, %vector.ph.preheader ]
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  %257 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03, i64 %index
  %wide.load = load <8 x i32>, ptr %257, align 64, !invariant.load !1, !noalias !6
  %258 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03, i64 %index
  %wide.load47 = load <8 x i32>, ptr %258, align 64, !invariant.load !1, !noalias !6
  %259 = add <8 x i32> %wide.load47, %wide.load
  %260 = lshr <8 x i32> %wide.load47, %broadcast.splat
  %261 = select i1 %shft.chk10, <8 x i32> %260, <8 x i32> zeroinitializer
  %262 = xor <8 x i32> %261, %259
  %263 = add <8 x i32> %262, %259
  %264 = shl <8 x i32> %262, %broadcast.splat38
  %265 = select i1 %shft.chk11, <8 x i32> %264, <8 x i32> zeroinitializer
  %266 = lshr <8 x i32> %262, %broadcast.splat40
  %267 = select i1 %shft.chk13, <8 x i32> %266, <8 x i32> zeroinitializer
  %268 = or <8 x i32> %265, %267
  %269 = xor <8 x i32> %268, %263
  %270 = add <8 x i32> %269, %263
  %271 = shl <8 x i32> %269, %broadcast.splat42
  %272 = select i1 %shft.chk14, <8 x i32> %271, <8 x i32> zeroinitializer
  %273 = lshr <8 x i32> %269, %broadcast.splat44
  %274 = select i1 %shft.chk16, <8 x i32> %273, <8 x i32> zeroinitializer
  %275 = or <8 x i32> %272, %274
  %276 = xor <8 x i32> %275, %270
  %277 = add <8 x i32> %270, %broadcast.splat46
  %278 = add <8 x i32> %277, %276
  %279 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03, i64 %index
  store <8 x i32> %278, ptr %279, align 64, !alias.scope !6
  %index.next = or disjoint i64 %index, 8
  %280 = getelementptr inbounds [512 x [512 x i32]], ptr %arg3, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03, i64 %index.next
  %wide.load.1 = load <8 x i32>, ptr %280, align 32, !invariant.load !1, !noalias !6
  %281 = getelementptr inbounds [512 x [512 x i32]], ptr %arg2, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03, i64 %index.next
  %wide.load47.1 = load <8 x i32>, ptr %281, align 32, !invariant.load !1, !noalias !6
  %282 = add <8 x i32> %wide.load47.1, %wide.load.1
  %283 = lshr <8 x i32> %wide.load47.1, %broadcast.splat
  %284 = select i1 %shft.chk10, <8 x i32> %283, <8 x i32> zeroinitializer
  %285 = xor <8 x i32> %284, %282
  %286 = add <8 x i32> %285, %282
  %287 = shl <8 x i32> %285, %broadcast.splat38
  %288 = select i1 %shft.chk11, <8 x i32> %287, <8 x i32> zeroinitializer
  %289 = lshr <8 x i32> %285, %broadcast.splat40
  %290 = select i1 %shft.chk13, <8 x i32> %289, <8 x i32> zeroinitializer
  %291 = or <8 x i32> %288, %290
  %292 = xor <8 x i32> %291, %286
  %293 = add <8 x i32> %292, %286
  %294 = shl <8 x i32> %292, %broadcast.splat42
  %295 = select i1 %shft.chk14, <8 x i32> %294, <8 x i32> zeroinitializer
  %296 = lshr <8 x i32> %292, %broadcast.splat44
  %297 = select i1 %shft.chk16, <8 x i32> %296, <8 x i32> zeroinitializer
  %298 = or <8 x i32> %295, %297
  %299 = xor <8 x i32> %298, %293
  %300 = add <8 x i32> %293, %broadcast.splat46
  %301 = add <8 x i32> %300, %299
  %302 = getelementptr inbounds [512 x [512 x i32]], ptr %arg4, i64 0, i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03, i64 %index.next
  store <8 x i32> %301, ptr %302, align 32, !alias.scope !6
  %index.next.1 = add nuw nsw i64 %index, 16
  %303 = icmp eq i64 %index.next.1, 512
  br i1 %303, label %broadcast_add_fusion.1.clone.loop_exit.dim.1.split, label %vector.body, !llvm.loop !19

broadcast_add_fusion.1.clone.loop_exit.dim.1.split: ; preds = %vector.body
  %invar.inc = add nuw nsw i64 %broadcast_add_fusion.1.clone.invar_address.dim.0.03, 1
  %exitcond12.not = icmp eq i64 %invar.inc, %up_dim_0
  br i1 %exitcond12.not, label %return, label %vector.ph, !llvm.loop !12

return:                                           ; preds = %broadcast_add_fusion.1.clone.loop_exit.dim.1.split, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split, %broadcast_add_fusion.1.clone.loop_exit.dim.1.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us.split.us.us, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 3}
!1 = !{}
!2 = !{i64 1048576}
!3 = !{i64 64}
!4 = !{i64 16}
!5 = !{i64 4}
!6 = !{!7}
!7 = !{!"result slice: {index:7, offset:3145792, size:1048576}", !8}
!8 = !{!"XLA host kernel broadcast_add_fusion.1.clone AA domain"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.isvectorized", i32 1}
!11 = !{!"llvm.loop.unroll.runtime.disable"}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.unroll.disable"}
!14 = distinct !{!14, !10, !11}
!15 = distinct !{!15, !10, !11}
!16 = distinct !{!16, !10, !11}
!17 = distinct !{!17, !10, !11}
!18 = distinct !{!18, !10, !11}
!19 = distinct !{!19, !10, !11}
