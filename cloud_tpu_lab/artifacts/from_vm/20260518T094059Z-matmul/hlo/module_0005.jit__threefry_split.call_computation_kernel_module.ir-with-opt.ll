; ModuleID = '__compute_module_call_computation_kernel_module'
source_filename = "__compute_module_call_computation_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @call_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
return:
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg19_gep = getelementptr i8, ptr %args, i64 304
  %arg19 = load ptr, ptr %arg19_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !2
  %arg20_gep = getelementptr i8, ptr %args, i64 320
  %arg20 = load ptr, ptr %arg20_gep, align 8, !invariant.load !1, !dereferenceable !3, !align !2
  %arg21_gep = getelementptr i8, ptr %args, i64 336
  %arg21 = load ptr, ptr %arg21_gep, align 8, !invariant.load !1, !dereferenceable !3, !align !2
  %arg22_gep = getelementptr i8, ptr %args, i64 352
  %arg22 = load ptr, ptr %arg22_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !2
  %arg23_gep = getelementptr i8, ptr %args, i64 368
  %arg23 = load ptr, ptr %arg23_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !2
  %arg24_gep = getelementptr i8, ptr %args, i64 384
  %arg24 = load ptr, ptr %arg24_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !2
  %arg25_gep = getelementptr i8, ptr %args, i64 400
  %arg25 = load ptr, ptr %arg25_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !2
  %arg26_gep = getelementptr i8, ptr %args, i64 416
  %arg26 = load ptr, ptr %arg26_gep, align 8, !invariant.load !1, !dereferenceable !3, !align !2
  %arg27_gep = getelementptr i8, ptr %args, i64 432
  %arg27 = load ptr, ptr %arg27_gep, align 8, !invariant.load !1, !dereferenceable !3, !align !2
  %arg28_gep = getelementptr i8, ptr %args, i64 448
  %arg28 = load ptr, ptr %arg28_gep, align 8, !invariant.load !1, !dereferenceable !6, !align !2
  %arg29_gep = getelementptr i8, ptr %args, i64 464
  %arg29 = load ptr, ptr %arg29_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !2
  %arg30_gep = getelementptr i8, ptr %args, i64 480
  %arg30 = load ptr, ptr %arg30_gep, align 8, !invariant.load !1, !dereferenceable !3, !align !2
  %arg31_gep = getelementptr i8, ptr %args, i64 496
  %arg31 = load ptr, ptr %arg31_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !2
  %arg32_gep = getelementptr i8, ptr %args, i64 512
  %arg32 = load ptr, ptr %arg32_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !2
  %arg34_gep = getelementptr i8, ptr %args, i64 544
  %arg34 = load ptr, ptr %arg34_gep, align 8, !invariant.load !1, !dereferenceable !3, !align !2
  %arg35_gep = getelementptr i8, ptr %args, i64 560
  %arg35 = load ptr, ptr %arg35_gep, align 8, !invariant.load !1, !dereferenceable !3, !align !2
  %arg36_gep = getelementptr i8, ptr %args, i64 576
  %arg36 = load ptr, ptr %arg36_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !2
  %arg38_gep = getelementptr i8, ptr %args, i64 608
  %arg38 = load ptr, ptr %arg38_gep, align 8, !invariant.load !1, !dereferenceable !3, !align !2
  %1 = load i32, ptr %arg34, align 64, !alias.scope !7, !noalias !10
  %2 = icmp slt i32 %1, 5
  %3 = zext i1 %2 to i8
  store i8 %3, ptr %arg28, align 64, !alias.scope !17, !noalias !18
  br i1 %2, label %while.2.body.i.lr.ph, label %while.1_computation.exit

while.2.body.i.lr.ph:                             ; preds = %return
  %4 = getelementptr inbounds nuw i8, ptr %arg29, i64 4
  %5 = getelementptr inbounds nuw i8, ptr %arg29, i64 8
  %6 = getelementptr inbounds nuw i8, ptr %arg29, i64 12
  %7 = getelementptr inbounds nuw i8, ptr %arg19, i64 8
  %8 = getelementptr inbounds nuw i8, ptr %arg19, i64 16
  %9 = getelementptr inbounds nuw i8, ptr %arg19, i64 24
  %10 = getelementptr inbounds nuw i8, ptr %arg19, i64 32
  %11 = getelementptr inbounds nuw i8, ptr %arg19, i64 40
  %12 = getelementptr inbounds nuw i8, ptr %arg19, i64 48
  %13 = getelementptr inbounds nuw i8, ptr %arg19, i64 56
  %14 = getelementptr inbounds nuw i8, ptr %arg23, i64 4
  %15 = getelementptr inbounds nuw i8, ptr %arg32, i64 4
  %16 = getelementptr inbounds nuw i8, ptr %arg22, i64 4
  %17 = getelementptr inbounds nuw i8, ptr %arg24, i64 4
  br label %while.2.body.i

while.2.body.i:                                   ; preds = %while.2.body.i.lr.ph, %while.2.body.i
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg36, ptr noundef nonnull align 64 dereferenceable(16) %arg25, i64 16, i1 false), !noalias !19
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg29, ptr noundef nonnull align 64 dereferenceable(16) %arg31, i64 16, i1 false), !noalias !19
  %18 = load i32, ptr %arg20, align 64, !noalias !19
  store i32 %18, ptr %arg27, align 64, !noalias !19
  %19 = load i32, ptr %arg30, align 64, !noalias !19
  store i32 %19, ptr %arg26, align 64, !noalias !19
  %20 = load i32, ptr %arg35, align 64, !noalias !19
  store i32 %20, ptr %arg21, align 64, !noalias !19
  %21 = load i64, ptr %arg24, align 64, !noalias !19
  store i64 %21, ptr %arg32, align 64, !noalias !19
  %22 = load i64, ptr %arg22, align 64, !noalias !19
  store i64 %22, ptr %arg23, align 64, !noalias !19
  %23 = load i32, ptr %arg34, align 64, !noalias !19
  store i32 %23, ptr %arg38, align 64, !noalias !19
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg25, ptr noundef nonnull align 64 dereferenceable(16) %arg29, i64 16, i1 false), !noalias !19
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 64 dereferenceable(16) %arg31, ptr noundef nonnull align 64 dereferenceable(16) %arg36, i64 16, i1 false), !noalias !19
  %24 = load i32, ptr %arg27, align 64, !noalias !19
  store i32 %24, ptr %arg30, align 64, !noalias !19
  %25 = load i32, ptr %arg21, align 64, !noalias !19
  store i32 %25, ptr %arg20, align 64, !noalias !19
  %26 = load i32, ptr %arg26, align 64, !noalias !19
  store i32 %26, ptr %arg35, align 64, !noalias !19
  %27 = load i32, ptr %arg29, align 64, !alias.scope !22, !noalias !24
  %shft.chk.i.i = icmp ult i32 %27, 32
  %28 = sub i32 32, %27
  %shft.chk1.i.i = icmp ult i32 %28, 32
  %29 = load i32, ptr %4, align 4, !alias.scope !22, !noalias !24
  %shft.chk2.i.i = icmp ult i32 %29, 32
  %30 = sub i32 32, %29
  %shft.chk4.i.i = icmp ult i32 %30, 32
  %31 = load i32, ptr %5, align 8, !alias.scope !22, !noalias !24
  %shft.chk5.i.i = icmp ult i32 %31, 32
  %32 = sub i32 32, %31
  %shft.chk7.i.i = icmp ult i32 %32, 32
  %33 = load i32, ptr %arg21, align 64, !alias.scope !34, !noalias !35
  %34 = load i32, ptr %arg23, align 64, !alias.scope !38, !noalias !39
  %35 = load i32, ptr %arg32, align 64, !alias.scope !40, !noalias !41
  %36 = add i32 %35, %34
  %37 = shl i32 %35, %27
  %38 = select i1 %shft.chk.i.i, i32 %37, i32 0
  %39 = lshr i32 %35, %28
  %40 = select i1 %shft.chk1.i.i, i32 %39, i32 0
  %41 = or i32 %40, %38
  %42 = xor i32 %41, %36
  %43 = add i32 %42, %36
  %44 = shl i32 %42, %29
  %45 = select i1 %shft.chk2.i.i, i32 %44, i32 0
  %46 = lshr i32 %42, %30
  %47 = select i1 %shft.chk4.i.i, i32 %46, i32 0
  %48 = or i32 %45, %47
  %49 = xor i32 %48, %43
  %50 = add i32 %49, %43
  %51 = shl i32 %49, %31
  %52 = select i1 %shft.chk5.i.i, i32 %51, i32 0
  %53 = lshr i32 %49, %32
  %54 = select i1 %shft.chk7.i.i, i32 %53, i32 0
  %55 = or i32 %52, %54
  %56 = xor i32 %55, %50
  %57 = add i32 %50, %33
  %58 = add i32 %57, %56
  store i32 %58, ptr %arg22, align 64, !alias.scope !42, !noalias !43
  %59 = load i32, ptr %14, align 4, !alias.scope !38, !noalias !39
  %60 = load i32, ptr %15, align 4, !alias.scope !40, !noalias !41
  %61 = add i32 %60, %59
  %62 = shl i32 %60, %27
  %63 = select i1 %shft.chk.i.i, i32 %62, i32 0
  %64 = lshr i32 %60, %28
  %65 = select i1 %shft.chk1.i.i, i32 %64, i32 0
  %66 = or i32 %65, %63
  %67 = xor i32 %66, %61
  %68 = add i32 %67, %61
  %69 = shl i32 %67, %29
  %70 = select i1 %shft.chk2.i.i, i32 %69, i32 0
  %71 = lshr i32 %67, %30
  %72 = select i1 %shft.chk4.i.i, i32 %71, i32 0
  %73 = or i32 %70, %72
  %74 = xor i32 %73, %68
  %75 = add i32 %74, %68
  %76 = shl i32 %74, %31
  %77 = select i1 %shft.chk5.i.i, i32 %76, i32 0
  %78 = lshr i32 %74, %32
  %79 = select i1 %shft.chk7.i.i, i32 %78, i32 0
  %80 = or i32 %77, %79
  %81 = xor i32 %80, %75
  %82 = add i32 %75, %33
  %83 = add i32 %82, %81
  store i32 %83, ptr %16, align 4, !alias.scope !42, !noalias !43
  %84 = load i32, ptr %6, align 4, !alias.scope !22, !noalias !24
  %shft.chk17.i.i = icmp ult i32 %84, 32
  %85 = sub i32 32, %84
  %shft.chk19.i.i = icmp ult i32 %85, 32
  %86 = load i32, ptr %arg26, align 64, !alias.scope !46, !noalias !47
  %87 = load i32, ptr %arg38, align 64, !alias.scope !48, !noalias !49
  %88 = add i32 %86, 1
  %89 = add i32 %88, %87
  %90 = add i32 %56, %50
  %91 = shl i32 %56, %84
  %92 = select i1 %shft.chk17.i.i, i32 %91, i32 0
  %93 = lshr i32 %56, %85
  %94 = select i1 %shft.chk19.i.i, i32 %93, i32 0
  %95 = or i32 %92, %94
  %96 = xor i32 %95, %90
  %97 = add i32 %89, %96
  store i32 %97, ptr %arg24, align 64, !alias.scope !51, !noalias !52
  %98 = add i32 %81, %75
  %99 = shl i32 %81, %84
  %100 = select i1 %shft.chk17.i.i, i32 %99, i32 0
  %101 = lshr i32 %81, %85
  %102 = select i1 %shft.chk19.i.i, i32 %101, i32 0
  %103 = or i32 %100, %102
  %104 = xor i32 %103, %98
  %105 = add i32 %89, %104
  store i32 %105, ptr %17, align 4, !alias.scope !51, !noalias !52
  %106 = add i32 %87, 1
  store i32 %106, ptr %arg34, align 64, !alias.scope !7, !noalias !53
  store ptr %arg34, ptr %arg19, align 64, !alias.scope !54, !noalias !55
  store ptr %arg22, ptr %7, align 8, !alias.scope !54, !noalias !55
  store ptr %arg24, ptr %8, align 16, !alias.scope !54, !noalias !55
  store ptr %arg35, ptr %9, align 8, !alias.scope !54, !noalias !55
  store ptr %arg30, ptr %10, align 32, !alias.scope !54, !noalias !55
  store ptr %arg20, ptr %11, align 8, !alias.scope !54, !noalias !55
  store ptr %arg31, ptr %12, align 16, !alias.scope !54, !noalias !55
  store ptr %arg25, ptr %13, align 8, !alias.scope !54, !noalias !55
  %107 = icmp slt i32 %106, 5
  %108 = zext i1 %107 to i8
  store i8 %108, ptr %arg28, align 64, !alias.scope !17, !noalias !18
  br i1 %107, label %while.2.body.i, label %while.1_computation.exit

while.1_computation.exit:                         ; preds = %while.2.body.i, %return
  ret ptr null
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 64}
!3 = !{i64 4}
!4 = !{i64 8}
!5 = !{i64 16}
!6 = !{i64 1}
!7 = !{!8}
!8 = !{!"buffer: {index:7, offset:768, size:4}", !9}
!9 = !{!"XLA global AA domain"}
!10 = !{!11, !12, !13, !15}
!11 = !{!"buffer: {index:6, offset:0, size:4}", !9}
!12 = !{!"buffer: {index:7, offset:64, size:1}", !9}
!13 = distinct !{!13, !14, !"while.2__1: %buffer_table"}
!14 = distinct !{!14, !"while.2__1"}
!15 = distinct !{!15, !16, !"while.1_computation: %buffer_table"}
!16 = distinct !{!16, !"while.1_computation"}
!17 = !{!12}
!18 = !{!11, !8, !13, !15}
!19 = !{!20, !15}
!20 = distinct !{!20, !21, !"while.2: %buffer_table"}
!21 = distinct !{!21, !"while.2"}
!22 = !{!23}
!23 = !{!"buffer: {index:7, offset:64, size:16}", !9}
!24 = !{!25, !26, !27, !28, !29, !30, !31, !32, !33, !20, !15}
!25 = !{!"buffer: {index:1, offset:0, size:16}", !9}
!26 = !{!"buffer: {index:7, offset:192, size:16}", !9}
!27 = !{!"buffer: {index:7, offset:256, size:8}", !9}
!28 = !{!"buffer: {index:7, offset:320, size:8}", !9}
!29 = !{!"buffer: {index:7, offset:384, size:8}", !9}
!30 = !{!"buffer: {index:7, offset:448, size:8}", !9}
!31 = !{!"buffer: {index:7, offset:512, size:4}", !9}
!32 = !{!"buffer: {index:7, offset:576, size:4}", !9}
!33 = !{!"buffer: {index:7, offset:640, size:4}", !9}
!34 = !{!33}
!35 = !{!23, !27, !28, !29, !36, !37, !20, !15}
!36 = !{!"buffer: {index:7, offset:832, size:4}", !9}
!37 = !{!"buffer: {index:7, offset:960, size:4}", !9}
!38 = !{!28}
!39 = !{!23, !27, !29, !30, !31, !32, !33, !20, !15}
!40 = !{!27}
!41 = !{!23, !28, !29, !30, !31, !32, !33, !20, !15}
!42 = !{!29}
!43 = !{!25, !44, !23, !26, !27, !28, !30, !33, !8, !36, !45, !37, !20, !15}
!44 = !{!"buffer: {index:7, offset:0, size:64}", !9}
!45 = !{!"buffer: {index:7, offset:896, size:4}", !9}
!46 = !{!31}
!47 = !{!23, !27, !28, !30, !32, !36, !45, !20, !15}
!48 = !{!32}
!49 = !{!50, !23, !27, !28, !30, !31, !8, !20, !15}
!50 = !{!"buffer: {index:0, offset:0, size:4}", !9}
!51 = !{!30}
!52 = !{!25, !44, !23, !26, !27, !28, !29, !31, !32, !8, !36, !45, !37, !20, !15}
!53 = !{!50, !25, !44, !26, !29, !30, !32, !36, !45, !37, !20, !15}
!54 = !{!44}
!55 = !{!25, !26, !29, !30, !8, !36, !45, !37, !20, !15}
